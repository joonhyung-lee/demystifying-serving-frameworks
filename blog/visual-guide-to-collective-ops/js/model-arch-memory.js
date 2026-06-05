/* Model Architectures — Memory estimation, panel renderer, vLLM internals.
 * Split from model-architectures.js for readability.
 * Depends on window.ArchLib (from model-arch-data.js).
 */
(function () {
  "use strict";
  var A = window.ArchLib = window.ArchLib || {};
  var escHtml = A.escHtml;

  // =========================================================================
  // Memory Estimation (from InferenceX)
  // =========================================================================
  
  var GB = 1e9;
  
  var GPU_DEVICES = [
    { id: "a100-40", label: "A100 40GB", vramGB: 40 },
    { id: "a100-80", label: "A100 80GB", vramGB: 80 },
    { id: "h100-80", label: "H100 80GB", vramGB: 80 },
    { id: "h200-141", label: "H200 141GB", vramGB: 141 },
  ];
  var WEIGHT_DTYPES = [
    { id: "bf16", label: "BF16 / FP16 (2B)", bytes: 2 },
    { id: "fp8", label: "FP8 (1B)", bytes: 1 },
    { id: "mxfp4", label: "MXFP4 (~0.5B)", bytes: 0.5 },
  ];
  var KV_DTYPES = [
    { id: "bf16", label: "BF16 (2B)", bytes: 2 },
    { id: "fp8", label: "FP8 (1B)", bytes: 1 },
  ];
  
  function normParams(p) {
    if (!p || p <= 0) return 0;
    return p < 1e6 ? p * 1e9 : p;
  }
  
  function defaultWeightDtypeId(arch) {
    var f = (arch.features || []).join(" ").toLowerCase();
    if (f.includes("mxfp4")) return "mxfp4";
    if (f.includes("fp8")) return "fp8";
    return "bf16";
  }
  
  function computeMemory(arch, cfg) {
    var isMoE2 = arch.architectureType === "moe";
    var device =
      GPU_DEVICES.find(function (d) {
        return d.id === cfg.deviceId;
      }) || GPU_DEVICES[2];
    var wBytes =
      (
        WEIGHT_DTYPES.find(function (d) {
          return d.id === cfg.weightDtypeId;
        }) || { bytes: 2 }
      ).bytes;
    var kvBytes =
      (
        KV_DTYPES.find(function (d) {
          return d.id === cfg.kvDtypeId;
        }) || { bytes: 2 }
      ).bytes;
  
    var tp = Math.max(1, Math.floor(cfg.tp));
    var dp = Math.max(1, Math.floor(cfg.dp));
    var totalGpus = dp * tp;
    var hidden = arch.hiddenSize || 0;
    var numLayers = arch.numLayers || 0;
    var numHeads = arch.numHeads || 1;
    var numKVHeads = arch.numKVHeads || numHeads;
    var headDim =
      arch.headDim || (numHeads ? Math.round(hidden / numHeads) : 0);
    var ffnDim = arch.ffnDim || 0;
    var denseLayers = isMoE2 ? arch.denseFFNLayers || 0 : 0;
    var moeLayers = Math.max(0, numLayers - denseLayers);
    var totalParams2 = normParams(arch.totalParams);
  
    // --- Weight decomposition (expert vs dense) ---
    // Expert weights: SwiGLU = 3 matrices of (hidden × ffnDim) per expert per layer.
    // Dense weights: everything else (attention, embeddings, shared expert, lm_head, dense FFN layers).
    var routedExperts = 0,
      routedExpertParams = 0;
    if (isMoE2) {
      routedExperts = arch.hasSharedExpert
        ? Math.max(0, (arch.numExperts || 1) - 1)
        : arch.numExperts || 0;
      routedExpertParams = routedExperts * moeLayers * 3 * hidden * ffnDim;
      routedExpertParams = Math.min(routedExpertParams, totalParams2 * 0.98);
    }
    var nonExpertParams = Math.max(totalParams2 - routedExpertParams, 0);
  
    // --- Per-GPU weight sharding ---
    // Dense (non-expert) weights: sharded by TP, replicated across DP.
    //   dense/GPU = dense_total / tp
    // Expert weights: in vLLM, MoE experts are always partitioned across ALL
    //   GPUs in the tp×dp mesh (via TP shard, EP dispatch, or DP's implicit
    //   expert-sharding when dp×tp fills the device count). Each GPU holds
    //   1/(tp×dp) of the total routed-expert parameters.
    //   expert/GPU = expert_total / (tp × dp)
    var nonExpertWeightB = (nonExpertParams * wBytes) / tp;
    var expertWeightB = (routedExpertParams * wBytes) / totalGpus;
    var weightB = nonExpertWeightB + expertWeightB;
  
    // --- CUDA graph + overhead estimation ---
    // vLLM captures CUDA graphs for the forward pass. The graph pool memory
    // depends on the MoE dispatch collective and its persistent buffers:
    //   EP off  → AllGather+ReduceScatter: transient buffers, small graph reserve (~1.2-3.3 GiB)
    //   EP on   → All-to-All: persistent per-peer staging buffers captured inside
    //             the graph, reserve grows with #EP peers (~1.8-7.2 GiB)
    // Base: ~1.2 GiB (tp-only, no DP communication).
    // DP>1 adds ~2 GiB for graph captures with multi-rank coordination.
    // EP on adds ~0.5 GiB per EP peer for All-to-All staging buffers.
    var cudagraphB;
    var epPeers = dp > 1 ? dp : 0;
    if (dp <= 1) {
      cudagraphB = 1.2 * GB;
    } else if (!cfg.epEnabled || !isMoE2) {
      // EP off: AllGather+ReduceScatter — smaller persistent workspace
      cudagraphB = (2.7 + 0.1 * dp) * GB;
    } else {
      // EP on: All-to-All — persistent per-peer dispatch/combine buffers
      cudagraphB = (1.8 + 0.7 * dp) * GB;
    }
    var overheadB = cudagraphB + 1.0 * GB;  // +1 GiB fixed (CUDA context, framework)
  
    // --- KV cache ---
    // vLLM budget: KV = (vram × gmu) − weights − overhead
    // MLA: stores one shared latent (kv_lora_rank + qk_rope_head_dim) per token
    //   per layer, replicated across TP ranks (kvShard=1). TP raises redundancy.
    // GQA: KV heads can be sharded across min(tp, numKVHeads) ranks.
    var isMLA = arch.attentionType === "MLA";
    var maxTokens = Math.max(1, cfg.maxNumSeqs) * Math.max(1, cfg.maxModelLen);
    var kvShard = 1;
    var kvPerTokenPerLayerB = 0;
    if (isMLA) {
      var kvLora = 512, ropeDim = 64;
      kvShard = 1;  // MLA latent replicated across TP — not sharded
      kvPerTokenPerLayerB = (kvLora + ropeDim) * kvBytes;
    } else {
      kvShard = Math.min(tp, Math.max(1, numKVHeads));
      var sw = arch.slidingWindow || 0;
      var hasAlt = Boolean(
        arch.alternatingLayers && arch.alternatingLayers.length > 0
      );
      kvPerTokenPerLayerB = 2 * numKVHeads * headDim * kvBytes;
    }
    var capacityB = device.vramGB * GB;
    var availForKV = Math.max(0, capacityB * cfg.gpuMemUtil - weightB - overheadB);
    // Theoretical max KV if all maxTokens are filled
    var kvMaxB;
    if (hasAlt && sw > 0) {
      var slidingLayers = Math.floor(numLayers / 2);
      var fullLayers = numLayers - slidingLayers;
      var perTokenFull = kvPerTokenPerLayerB * fullLayers;
      var perTokenSliding = kvPerTokenPerLayerB * slidingLayers;
      kvMaxB = (perTokenFull + perTokenSliding) * maxTokens / kvShard;
    } else {
      kvMaxB = kvPerTokenPerLayerB * numLayers * maxTokens / kvShard;
    }
    var kvB = Math.min(kvMaxB, availForKV);
    var perTokenKvB = kvPerTokenPerLayerB * numLayers / kvShard;
    var kvTokens = perTokenKvB > 0 ? Math.floor(kvB / perTokenKvB) : 0;
    var totalB = weightB + kvB + overheadB;
  
    return {
      totalGpus: totalGpus,
      nonExpertWeightB: nonExpertWeightB,
      expertWeightB: expertWeightB,
      weightB: weightB,
      kvB: kvB,
      kvMaxB: kvMaxB,
      availForKV: availForKV,
      overheadB: overheadB,
      cudagraphB: cudagraphB,
      totalB: totalB,
      capacityB: capacityB,
      utilPct: (totalB / capacityB) * 100,
      fits: totalB <= capacityB,
      routedExperts: routedExperts,
      headDim: headDim,
      kvShard: kvShard,
      perTokenKvKB: perTokenKvB / 1024,
      kvTokens: kvTokens,
      maxTokens: maxTokens,
      isMoE: isMoE2,
      isMLA: isMLA,
      epPeers: epPeers,
    };
  }
  
  function fmtGB(bytes) {
    var gb = bytes / GB;
    if (gb >= 100) return gb.toFixed(0) + " GB";
    if (gb >= 10) return gb.toFixed(1) + " GB";
    return gb.toFixed(2) + " GB";
  }
  
  // =========================================================================
  // Memory Panel HTML renderer
  // =========================================================================
  
  var MEM_SEGMENTS = [
    { key: "nonExpert", label: "Weights \u00B7 dense", color: "#d97706" },
    { key: "expert", label: "Weights \u00B7 experts", color: "#9333ea" },
    { key: "kv", label: "KV cache", color: "#0284c7" },
    { key: "cudagraph", label: "CUDA graph", color: "#6366f1" },
    { key: "fixed", label: "Framework overhead", color: "#94a3b8" },
  ];
  
  function renderMemoryPanel(container, arch, cfg) {
    var m = computeMemory(arch, cfg);
    var barH = 200;
    var scaleMax = Math.max(m.capacityB, m.totalB) * 1.04;
    var px = function (bytes) {
      return (bytes / scaleMax) * barH;
    };
    var capY = barH - px(m.capacityB);
    var segValues = {
      nonExpert: m.nonExpertWeightB,
      expert: m.expertWeightB,
      kv: m.kvB,
      cudagraph: m.cudagraphB,
      fixed: m.overheadB - m.cudagraphB,
    };
    var acc = 0;
    var segRects = MEM_SEGMENTS.filter(function (s) {
      return segValues[s.key] > 0;
    }).map(function (s) {
      var h = px(segValues[s.key]);
      var yTop = barH - acc - h;
      acc += h;
      return {
        key: s.key,
        label: s.label,
        color: s.color,
        h: h,
        yTop: yTop,
        bytes: segValues[s.key],
      };
    });
  
    var fitsColor = m.fits ? "#16a34a" : "#dc2626";
    var dotCount = Math.min(m.totalGpus, 64);
  
    var segSvg = segRects
      .map(function (s) {
        return (
          '<rect x="20" y="' +
          s.yTop +
          '" width="44" height="' +
          Math.max(0, s.h) +
          '" fill="' +
          s.color +
          '" opacity="0.9"><title>' +
          escHtml(s.label + ": " + fmtGB(s.bytes)) +
          "</title></rect>"
        );
      })
      .join("");
  
    var breakdownHtml = segRects
      .map(function (s) {
        return (
          '<div class="arch-mem-row"><span class="arch-mem-dot" style="background:' +
          s.color +
          '"></span><span class="arch-mem-label">' +
          escHtml(s.label) +
          '</span><span class="arch-mem-val">' +
          fmtGB(s.bytes) +
          "</span></div>"
        );
      })
      .join("");
  
    var dots = "";
    for (var i = 0; i < dotCount; i++) {
      dots +=
        '<span class="arch-mesh-dot" style="background:' +
        fitsColor +
        '"></span>';
    }
    if (m.totalGpus > dotCount)
      dots +=
        '<span style="font-size:.7rem;color:#6b7280">+' +
        (m.totalGpus - dotCount) +
        "</span>";
  
    container.innerHTML =
      '<div class="arch-mem-header"><strong>Memory Allocation</strong> <span class="arch-badge-sm">per device</span></div>' +
      '<div class="arch-mem-body">' +
      '<svg width="92" height="' +
      (barH + 16) +
      '" style="flex-shrink:0;overflow:visible">' +
      '<rect x="20" y="0" width="44" height="' +
      barH +
      '" rx="5" fill="#e5e7eb" opacity="0.4"/>' +
      segSvg +
      '<line x1="12" y1="' +
      capY +
      '" x2="72" y2="' +
      capY +
      '" stroke="' +
      fitsColor +
      '" stroke-width="1.5" stroke-dasharray="4 3"/>' +
      '<text x="74" y="' +
      (capY + 3) +
      '" font-size="9" fill="#6b7280">' +
      (m.capacityB / GB) +
      "GB</text></svg>" +
      '<div class="arch-mem-detail">' +
      '<div class="arch-mem-total"><span>Per-GPU total</span><strong>' +
      fmtGB(m.totalB) +
      "</strong></div>" +
      '<div class="arch-mem-bar-track"><div class="arch-mem-bar-fill" style="width:' +
      Math.min(100, m.utilPct) +
      "%;background:" +
      fitsColor +
      '"></div></div>' +
      '<span class="arch-badge-sm" style="color:' +
      fitsColor +
      ";border-color:" +
      fitsColor +
      '55">' +
      (m.fits ? "Fits" : "Over capacity") +
      " \u00B7 " +
      m.utilPct.toFixed(0) +
      "%</span>" +
      breakdownHtml +
      "</div></div>" +
      '<div class="arch-mem-mesh"><span>Device mesh \u00B7 DP ' +
      cfg.dp +
      " \u00D7 TP " +
      cfg.tp +
      " = " +
      m.totalGpus +
      " GPU" +
      (m.totalGpus > 1 ? "s" : "") +
      (arch.architectureType === "moe" && cfg.epEnabled ? " \u00B7 EP on" : "") +
      '</span><div class="arch-mesh-dots">' +
      dots +
      "</div></div>" +
      '<p class="arch-mem-note">Estimate \u00B7 dense \u00F7TP' +
      (m.isMoE ? ", experts \u00F7(TP\u00D7DP)=" + m.totalGpus : "") +
      ". " +
      (m.isMoE && cfg.dp > 1
        ? (cfg.epEnabled
          ? "EP on: All-to-All dispatch (per-peer buffers in CUDA graph). "
          : "EP off: AllGather+ReduceScatter (transient buffers). ")
        : "") +
      "KV: " + m.perTokenKvKB.toFixed(1) + " KB/token" +
      (m.isMLA ? " (MLA latent, replicated across TP)" : " \u00F7" + m.kvShard) +
      ". " + m.kvTokens.toLocaleString() + " tokens fit" +
      " (" + fmtGB(m.availForKV) + " avail). " +
      "CUDA graph \u2248" + fmtGB(m.cudagraphB) + " + 1 GB fixed.</p>" +
      // --- vLLM Internals: collapsible source reference ---
      buildVllmInternals(arch, cfg, m);
  }
  
  function buildVllmInternals(arch, cfg, m) {
    var isMoE = arch.architectureType === "moe";
    var dp = cfg.dp, tp = cfg.tp;
  
    // Section 1: Memory budget identity
    var budgetCode =
      "# vllm/v1/worker/gpu_worker.py — determine_available_memory()\n" +
      "\n" +
      "profile_result.non_kv_cache_memory = (\n" +
      "    profile_result.non_torch_increase\n" +
      "    + profile_result.torch_peak_increase    # activation peak\n" +
      "    + profile_result.weights_memory          # model weights\n" +
      ")\n" +
      "\n" +
      "self.available_kv_cache_memory_bytes = (\n" +
      "    self.requested_memory                    # vram × gpu_memory_utilization\n" +
      "    - profile_result.non_kv_cache_memory     # weights + activations\n" +
      "    - cudagraph_memory_estimate              # captured CUDA graph pool\n" +
      ")\n" +
      "\n" +
      "# Your config:\n" +
      "#   requested = " + (m.capacityB * cfg.gpuMemUtil / GB).toFixed(1) + " GiB  (" + (m.capacityB / GB).toFixed(0) + " × " + cfg.gpuMemUtil + ")\n" +
      "#   weights   = " + fmtGB(m.weightB) + "\n" +
      "#   cudagraph = " + fmtGB(m.cudagraphB) + "\n" +
      "#   → KV avail = " + fmtGB(m.availForKV);
  
    // Section 2: Weight sharding
    var shardCode =
      "# Weight sharding per GPU\n" +
      "#\n" +
      "# Dense (attention, embeddings, shared expert, lm_head):\n" +
      "#   Sharded by TP, replicated across DP.\n" +
      "#   dense/GPU = dense_total / tp\n" +
      "#            = " + fmtGB(m.nonExpertWeightB) + "  (" + fmtGB(m.nonExpertWeightB * tp) + " / tp=" + tp + ")\n";
  
    if (isMoE) {
      shardCode +=
        "#\n" +
        "# Routed experts (SwiGLU: gate + up + down per expert per layer):\n" +
        "#   Partitioned across ALL GPUs in the tp×dp mesh.\n" +
        "#   expert/GPU = expert_total / (tp × dp)\n" +
        "#             = " + fmtGB(m.expertWeightB) + "  (" + fmtGB(m.expertWeightB * tp * dp) + " / " + (tp * dp) + ")\n" +
        "#\n" +
        "# Why experts ÷ (tp×dp), not ÷ tp?\n" +
        "#   vLLM always partitions MoE experts across the full device mesh.\n" +
        "#   With dp>1, each GPU holds 1/(tp×dp) of routed expert params,\n" +
        "#   and uses AllGather+RS (EP off) or All-to-All (EP on) to route\n" +
        "#   tokens to the correct expert's owner GPU.";
    }
  
    // Section 3: MoE dispatch (only for MoE + dp>1)
    var dispatchCode = "";
    if (isMoE && dp > 1) {
      if (!cfg.epEnabled) {
        dispatchCode =
          "# vllm/distributed/device_communicators/all2all.py\n" +
          "# EP off → AllGather + ReduceScatter (fallback)\n" +
          "\n" +
          "# vllm/model_executor/layers/fused_moe/all2all_utils.py:110\n" +
          "if not moe.moe_parallel_config.use_all2all_kernels:\n" +
          "    if moe.moe_parallel_config.dp_size > 1:\n" +
          "        logger.info_once(\n" +
          "            \"Detected DP deployment with no --enable-expert-parallel.\"\n" +
          "            \" Falling back to AllGather+ReduceScatter dispatch/combine.\"\n" +
          "        )\n" +
          "\n" +
          "# class AgRsAll2AllManager:\n" +
          "def dispatch(self, hidden_states, topk_weights, topk_ids, ...):\n" +
          "    gathered = dist_group.all_gatherv(tensors, dim=0, sizes=sizes)\n" +
          "    #   → transient buffer: all DP ranks' tokens materialized\n" +
          "    #   → reused by activation memory, small CUDA graph footprint\n" +
          "\n" +
          "def combine(self, hidden_states, ...):\n" +
          "    hidden = dist_group.reduce_scatterv(hidden, dim=0, sizes=sizes)\n" +
          "    #   → result scattered back to owner ranks";
      } else {
        dispatchCode =
          "# vllm/distributed/device_communicators/all2all.py\n" +
          "# EP on → All-to-All (per-peer routing)\n" +
          "\n" +
          "# Each token is routed only to the rank that owns its selected expert.\n" +
          "# Symmetric per-peer send/recv staging buffers are allocated,\n" +
          "# sized for worst-case (max_num_tokens × topk × hidden) per peer.\n" +
          "#\n" +
          "# These buffers are PERSISTENT and captured inside the CUDA graph\n" +
          "# → measured by profile_cudagraph_memory() via mem_before − mem_after\n" +
          "# → subtracted from KV budget\n" +
          "\n" +
          "# Per-peer buffer ≈ max_tokens × topk × hidden × dtype_bytes / dp\n" +
          "# With " + dp + " EP peers → CUDA graph reserve ≈ " + fmtGB(m.cudagraphB);
      }
    }
  
    // Section 4: CUDA graph profiling
    var cudagraphCode =
      "# vllm/v1/worker/gpu_model_runner.py — profile_cudagraph_memory()\n" +
      "#\n" +
      "# Does NOT estimate from a formula — actually captures sample graphs\n" +
      "# and measures GPU free-memory drop:\n" +
      "\n" +
      "for i, desc in enumerate(profile_descs):\n" +
      "    mem_before = torch.cuda.mem_get_info()[0]\n" +
      "    self._warmup_and_capture(desc, cudagraph_runtime_mode=mode)\n" +
      "    torch.accelerator.synchronize()\n" +
      "    free_after = torch.cuda.mem_get_info()[0]\n" +
      "    mem_samples.append(mem_before - free_after)\n" +
      "\n" +
      "# mem_before − free_after captures whatever persistent buffers the\n" +
      "# kernels inside the graph allocate — including MoE dispatch workspace.\n" +
      "# → EP on's All-to-All staging buffers are measured here.";
  
    // Section 5: KV cache
    var kvCode;
    if (m.isMLA) {
      var mlaKvBytes = (KV_DTYPES.find(function(d){return d.id===cfg.kvDtypeId}) || {bytes:2}).bytes;
      var mlaPerLayer = (512 + 64) * mlaKvBytes;
      var mlaAllLayersKB = (mlaPerLayer * arch.numLayers / 1024).toFixed(1);
      kvCode =
        "# KV cache — Multi-head Latent Attention (MLA)\n" +
        "#\n" +
        "# MLA stores a compressed latent per token per layer:\n" +
        "#   per_layer = (kv_lora_rank + qk_rope_head_dim) × kv_dtype_bytes\n" +
        "#             = (512 + 64) × " + mlaKvBytes + "B = " + mlaPerLayer.toLocaleString() + " bytes/token/layer\n" +
        "#\n" +
        "# × " + arch.numLayers + " layers  = " + mlaAllLayersKB + " KB/token\n" +
        "# ÷ kvShard(1) = " + m.perTokenKvKB.toFixed(1) + " KB/token/GPU  (MLA latent is replicated, not sharded)\n" +
        "#\n" +
        "# MLA latent is REPLICATED across TP ranks.\n" +
        "# → Raising TP stores the same tokens tp times (KV-inefficient).\n" +
        "# → Raising DP gives dp independent KV pools, each with unique tokens.\n" +
        "#\n" +
        "# tokens_that_fit = avail_kv / per_token_kv_per_gpu\n" +
        "#                 = " + fmtGB(m.availForKV) + " / " + m.perTokenKvKB.toFixed(1) + " KB\n" +
        "#                 = " + m.kvTokens.toLocaleString() + " tokens/replica";
    } else {
      var perLayerBytes = 2 * (arch.numKVHeads || 1) * m.headDim * (KV_DTYPES.find(function(d){return d.id===cfg.kvDtypeId}) || {bytes:2}).bytes;
      var allLayersKB = (perLayerBytes * arch.numLayers / 1024).toFixed(1);
      kvCode =
        "# KV cache — Grouped Query Attention (GQA)\n" +
        "#\n" +
        "# Per-token per-layer = 2 × num_kv_heads × head_dim × kv_dtype_bytes\n" +
        "#                     = 2 × " + (arch.numKVHeads || "?") + " × " + m.headDim + " × " + (KV_DTYPES.find(function(d){return d.id===cfg.kvDtypeId}) || {bytes:2}).bytes + "B\n" +
        "#                     = " + perLayerBytes.toLocaleString() + " bytes/token/layer\n" +
        "#\n" +
        "# × " + arch.numLayers + " layers       = " + allLayersKB + " KB/token (total)\n" +
        "# ÷ kvShard(" + m.kvShard + ")    = " + m.perTokenKvKB.toFixed(1) + " KB/token/GPU\n" +
        "#   where kvShard = min(tp=" + tp + ", num_kv_heads=" + (arch.numKVHeads || "?") + ") = " + m.kvShard + "\n" +
        "#\n" +
        "# tokens_that_fit = avail_kv / per_token_kv_per_gpu\n" +
        "#                 = " + fmtGB(m.availForKV) + " / " + m.perTokenKvKB.toFixed(1) + " KB\n" +
        "#                 = " + m.kvTokens.toLocaleString() + " tokens/replica";
    }
  
    // Build the collapsible sections
    // GitHub permalinks pinned to vllm v0.20.1 tag
    var VLLM_BASE = "https://github.com/vllm-project/vllm/blob/v0.20.1";
    var sections = [
      { id: "budget", title: "Memory Budget Identity", code: budgetCode,
        file: "vllm/v1/worker/gpu_worker.py",
        ghUrl: VLLM_BASE + "/vllm/v1/worker/gpu_worker.py#L332-L456" },
      { id: "weights", title: "Weight Sharding", code: shardCode,
        file: "weight decomposition", ghUrl: null },
    ];
    if (dispatchCode) {
      sections.push({
        id: "dispatch",
        title: "MoE Dispatch (" + (cfg.epEnabled ? "All-to-All" : "AllGather+RS") + ")",
        code: dispatchCode,
        file: "vllm/…/all2all_utils.py",
        ghUrl: cfg.epEnabled
          ? VLLM_BASE + "/vllm/distributed/device_communicators/all2all.py#L84-L160"
          : VLLM_BASE + "/vllm/model_executor/layers/fused_moe/all2all_utils.py#L110-L130"
      });
    }
    sections.push(
      { id: "cudagraph", title: "CUDA Graph Profiling", code: cudagraphCode,
        file: "vllm/v1/worker/gpu_model_runner.py",
        ghUrl: VLLM_BASE + "/vllm/v1/worker/gpu_model_runner.py#L5949-L6010" },
      { id: "kvcache", title: "KV Cache (" + (m.isMLA ? "MLA" : "GQA") + ")", code: kvCode,
        file: "vllm/v1/worker/gpu_worker.py",
        ghUrl: VLLM_BASE + "/vllm/v1/worker/gpu_worker.py#L440-L490" }
    );
  
    var html = '<div class="arch-internals">' +
      '<button type="button" class="arch-internals-toggle" onclick="this.parentElement.classList.toggle(\'open\')">' +
      '<span class="arch-internals-icon">&#9662;</span> How vLLM Allocates Memory' +
      '</button>' +
      '<div class="arch-internals-body">';
  
    sections.forEach(function (s) {
      var ghLink = s.ghUrl
        ? '<a href="' + s.ghUrl + '" target="_blank" rel="noopener" class="arch-gh-link">Source &#8599;</a>'
        : '';
      html +=
        '<details class="arch-code-section">' +
        '<summary><code>' + escHtml(s.file) + '</code> — ' + escHtml(s.title) + ghLink + '</summary>' +
        '<pre class="arch-code-block">' + escHtml(s.code) + '</pre>' +
        '</details>';
    });
  
    html += '</div></div>';
    return html;
  }

  // Exports
  A.GB = GB;
  A.GPU_DEVICES = GPU_DEVICES;
  A.WEIGHT_DTYPES = WEIGHT_DTYPES;
  A.KV_DTYPES = KV_DTYPES;
  A.computeMemory = computeMemory;
  A.fmtGB = fmtGB;
  A.defaultWeightDtypeId = defaultWeightDtypeId;
  A.renderMemoryPanel = renderMemoryPanel;
  A.buildVllmInternals = buildVllmInternals;
  A.MEM_SEGMENTS = MEM_SEGMENTS;
})();
