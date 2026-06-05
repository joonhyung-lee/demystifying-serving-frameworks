/* Model Architectures — Data, utilities, sub-block generators, color palette.
 * Split from model-architectures.js for readability.
 * Exports via window.ArchLib namespace.
 */
(function () {
  "use strict";
  var A = window.ArchLib = window.ArchLib || {};

  // =========================================================================
  // Model Architecture Data (from InferenceX model-architectures.ts)
  // =========================================================================
  
  const MODEL_ARCHITECTURES = [
    // --- 1. gpt-oss-120b ---
    {
      id: "gpt-oss-120b",
      name: "gpt-oss 120B",
      totalParams: 120,
      activeParams: 5,
      architectureType: "moe",
      attentionType: "AlternatingSinkGQA",
      numLayers: 36,
      hiddenSize: 2880,
      numHeads: 64,
      numKVHeads: 8,
      headDim: 64,
      vocabSize: 201088,
      ffnDim: 2880,
      numExperts: 128,
      activeExperts: 4,
      hasSharedExpert: false,
      alternatingLayers: [
        {
          label: "Sliding Attention + Sink",
          description:
            "GQA with 128-token sliding window and learnable attention sink tokens",
          count: 18,
          colorKey: "attention",
        },
        {
          label: "Causal Grouped Query Attention",
          description:
            "Standard GQA with full causal masking over entire context",
          count: 18,
          colorKey: "norm",
        },
      ],
      slidingWindow: 128,
      contextWindow: 131072,
      features: [
        "Alternating Sliding/Full Attention",
        "Attention Sink Tokens",
        "YaRN RoPE (factor=32)",
        "MXFP4 Quantization",
      ],
      releaseDate: "2025-06-13",
      developer: "OpenAI",
      sourceUrl: "https://huggingface.co/openai/gpt-oss-120b",
      ckptDtype: "mxfp4",
      ckptBytes: 0.5,
      defaultDp: 1, defaultTp: 1,
    },
    // --- 2. MiniMax M2.5 ---
    {
      id: "minimax-m2.5",
      name: "MiniMax M2.5",
      totalParams: 230,
      activeParams: 10,
      architectureType: "moe",
      attentionType: "GQA",
      attentionExpandable: false,
      numLayers: 62,
      hiddenSize: 3072,
      numHeads: 48,
      numKVHeads: 8,
      headDim: 128,
      vocabSize: 200064,
      ffnDim: 1536,
      numExperts: 256,
      activeExperts: 8,
      hasSharedExpert: false,
      contextWindow: 196608,
      features: [
        "GQA with QK Norm",
        "RoPE",
        "Multi-Token Prediction (3 modules)",
        "FP8 Quantization",
      ],
      releaseDate: "2025-10-25",
      developer: "MiniMax",
      sourceUrl: "https://huggingface.co/MiniMaxAI/MiniMax-M2",
      ckptDtype: "fp8",
      ckptBytes: 1,
      defaultDp: 8, defaultTp: 1,
    },
    // --- 3. DeepSeek V3.1 ---
    {
      id: "deepseek-v3.1",
      name: "DeepSeek V3.1",
      totalParams: 671,
      activeParams: 37,
      architectureType: "moe",
      attentionType: "MLA",
      numLayers: 61,
      hiddenSize: 7168,
      numHeads: 128,
      vocabSize: 129280,
      ffnDim: 2048,
      numExperts: 257,
      activeExperts: 8,
      hasSharedExpert: true,
      denseFFNLayers: 3,
      denseFFNDim: 18432,
      contextWindow: 163840,
      features: [
        "Multi-head Latent Attention",
        "Auxiliary-loss-free Load Balancing",
        "Multi-Token Prediction",
        "FP8 Training (E4M3)",
        "YaRN RoPE (factor=40)",
      ],
      releaseDate: "2025-05-20",
      developer: "DeepSeek",
      sourceUrl: "https://huggingface.co/deepseek-ai/DeepSeek-V3.1",
      ckptDtype: "fp8",
      ckptBytes: 1,
      defaultDp: 1, defaultTp: 8,
    },
    // --- 4. DeepSeek R1 ---
    {
      id: "deepseek-r1",
      name: "DeepSeek R1 (0528)",
      totalParams: 671,
      activeParams: 37,
      architectureType: "moe",
      attentionType: "MLA",
      numLayers: 61,
      hiddenSize: 7168,
      numHeads: 128,
      vocabSize: 129280,
      ffnDim: 2048,
      numExperts: 257,
      activeExperts: 8,
      hasSharedExpert: true,
      denseFFNLayers: 3,
      denseFFNDim: 18432,
      contextWindow: 128000,
      features: [
        "Multi-head Latent Attention",
        "Auxiliary-loss-free Load Balancing",
        "Multi-Token Prediction",
      ],
      releaseDate: "2025-05-28",
      developer: "DeepSeek",
      sourceUrl: "https://huggingface.co/deepseek-ai/DeepSeek-R1-0528",
      ckptDtype: "bf16",
      ckptBytes: 2,
      defaultDp: 1, defaultTp: 8,
    },
    // --- 5. Kimi K2.5 ---
    {
      id: "kimi-k2.5",
      name: "Kimi K2.5",
      totalParams: 1000,
      activeParams: 32,
      architectureType: "moe",
      attentionType: "MLA",
      numLayers: 61,
      hiddenSize: 7168,
      numHeads: 64,
      vocabSize: 163840,
      ffnDim: 2048,
      numExperts: 385,
      activeExperts: 8,
      hasSharedExpert: true,
      denseFFNLayers: 1,
      denseFFNDim: 18432,
      contextWindow: 262144,
      features: [
        "Multi-head Latent Attention",
        "DeepSeek-style MoE",
        "YaRN RoPE",
      ],
      releaseDate: "2026-01-27",
      developer: "Moonshot AI",
      sourceUrl: "https://huggingface.co/moonshotai/Kimi-K2.5",
      ckptDtype: "bf16",
      ckptBytes: 2,
      defaultDp: 1, defaultTp: 8,
    },
    // --- 6. Llama 3.3 70B ---
    {
      id: "llama-3.3-70b",
      name: "Llama 3.3 70B",
      totalParams: 70,
      activeParams: 70,
      architectureType: "dense",
      attentionType: "GQA",
      numLayers: 80,
      hiddenSize: 8192,
      numHeads: 64,
      numKVHeads: 8,
      vocabSize: 128256,
      ffnDim: 28672,
      contextWindow: 128000,
      features: ["Grouped Query Attention", "RoPE"],
      releaseDate: "2024-12-06",
      developer: "Meta",
      sourceUrl: "https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct",
      ckptDtype: "fp8",
      ckptBytes: 1,
      defaultDp: 1, defaultTp: 1,
    },
    // --- 7. Llama 3.1 70B ---
    {
      id: "llama-3.1-70b",
      name: "Llama 3.1 70B",
      totalParams: 70,
      activeParams: 70,
      architectureType: "dense",
      attentionType: "GQA",
      numLayers: 80,
      hiddenSize: 8192,
      numHeads: 64,
      numKVHeads: 8,
      vocabSize: 128256,
      ffnDim: 28672,
      contextWindow: 128000,
      features: ["Grouped Query Attention", "RoPE"],
      releaseDate: "2024-07-23",
      developer: "Meta",
      sourceUrl: "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct",
      ckptDtype: "fp8",
      ckptBytes: 1,
      defaultDp: 1, defaultTp: 1,
    },
  ];
  
  // =========================================================================
  // Utility helpers
  // =========================================================================
  
  function formatParamCount(params) {
    if (params >= 1000) return (params / 1000).toFixed(1) + "T";
    return params + "B";
  }
  function formatContextWindow(tokens) {
    if (tokens >= 1000000) return (tokens / 1000000).toFixed(0) + "M";
    return (tokens / 1000).toFixed(0) + "K";
  }
  function getAttentionLabel(type) {
    const map = {
      MHA: "Multi-Head Attention",
      GQA: "Grouped Query Attention",
      MLA: "Multi-head Latent Attention",
      Linear: "Linear Attention",
      Hybrid: "Hybrid Attention",
      AlternatingSinkGQA: "Alternating Sink/Full GQA",
    };
    return map[type] || type;
  }
  function escHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }
  
  // =========================================================================
  // Per-layer weight memory calculation (checkpoint dtype)
  // =========================================================================
  
  /** Format bytes to human-readable GB/MB string */
  function fmtBytes(bytes) {
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + " GB";
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + " MB";
    if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + " KB";
    return bytes + " B";
  }
  
  /**
   * Compute per-component weight memory sizes for a model architecture.
   * All sizes in bytes, based on ckptBytes (bytes per parameter).
   */
  function computeLayerMemory(arch) {
    var B = arch.ckptBytes || 2; // default bf16
    var h = arch.hiddenSize || 0;
    var vocab = arch.vocabSize || 0;
    var numHeads = arch.numHeads || 1;
    var numKVHeads = arch.numKVHeads || numHeads;
    var headDim = arch.headDim || (numHeads ? Math.round(h / numHeads) : 128);
    var ffnDim = arch.ffnDim || 0;
    var isMoE = arch.architectureType === "moe";
  
    // Embedding: vocab × hidden
    var embedding = vocab * h * B;
  
    // Attention per layer: Q + K + V + O projections
    var qParams = h * numHeads * headDim;
    var kParams = h * numKVHeads * headDim;
    var vParams = h * numKVHeads * headDim;
    var oParams = numHeads * headDim * h;
    var attentionPerLayer = (qParams + kParams + vParams + oParams) * B;
  
    // Dense FFN per layer (SwiGLU: gate + up + down = 3 × h × ffnDim)
    var denseFfnDim = isMoE ? (arch.denseFFNDim || ffnDim) : ffnDim;
    var denseFfnPerLayer = 3 * h * denseFfnDim * B;
  
    // MoE expert FFN: all experts combined per layer
    var numExperts = arch.numExperts || 0;
    var allExpertsPerLayer = numExperts * 3 * h * ffnDim * B;
  
    // Shared expert (1 expert)
    var sharedExpertPerLayer = arch.hasSharedExpert ? (3 * h * ffnDim * B) : 0;
  
    // Router: h × numExperts
    var routerPerLayer = isMoE ? (h * numExperts * B) : 0;
  
    // RMSNorm: 2 × h per layer (pre-attn + pre-FFN)
    var normPerLayer = 2 * h * B;
  
    // Output head: vocab × hidden (often tied w/ embedding)
    var outputHead = vocab * h * B;
  
    // Layer counts
    var numLayers = arch.numLayers || 0;
    var denseFFNLayers = arch.denseFFNLayers || 0;
    var moeLayers = isMoE ? (numLayers - denseFFNLayers) : 0;
    var denseLayers = isMoE ? denseFFNLayers : numLayers;
  
    // Total — use known totalParams for accuracy (component sum can diverge
    // for MLA models where our Q/K/V formula doesn't match the actual
    // compressed projections, and for tied embedding/output-head weights).
    var total = arch.totalParams * 1e9 * B;
  
    return {
      embedding: embedding,
      attentionPerLayer: attentionPerLayer,
      denseFfnPerLayer: denseFfnPerLayer,
      allExpertsPerLayer: allExpertsPerLayer,
      sharedExpertPerLayer: sharedExpertPerLayer,
      routerPerLayer: routerPerLayer,
      normPerLayer: normPerLayer,
      outputHead: outputHead,
      total: total,
      // For the collapsed transformer block label
      transformerPerLayer: attentionPerLayer + normPerLayer +
        (isMoE ? (allExpertsPerLayer + sharedExpertPerLayer + routerPerLayer)
               : denseFfnPerLayer),
      denseTransformerPerLayer: attentionPerLayer + normPerLayer + denseFfnPerLayer,
    };
  }
  
  // =========================================================================
  // Sub-block flow data generators (SwiGLU FFN + GQA Attention)
  // =========================================================================
  
  function getFFNSubBlocks(arch, opts) {
    const ffnDim =
      opts && opts.useDenseFFNDim && arch.denseFFNDim
        ? arch.denseFFNDim
        : arch.ffnDim;
    const hiddenSize = arch.hiddenSize;
    return {
      layout: "parallel",
      leftPath: [
        {
          name: "Gate Projection",
          detail: ffnDim ? "\u2192 " + ffnDim.toLocaleString() : undefined,
          type: "projection",
        },
        {
          name: "SiLU Activation",
          detail: "Applied to gate output",
          type: "activation",
        },
      ],
      rightPath: [
        {
          name: "Up Projection",
          detail: ffnDim ? "\u2192 " + ffnDim.toLocaleString() : undefined,
          type: "projection",
        },
      ],
      mergeBlocks: [
        { name: "\u2297", circleSymbol: "\u00D7", type: "operation" },
        {
          name: "Down Projection",
          detail: hiddenSize
            ? "\u2192 " + hiddenSize.toLocaleString()
            : undefined,
          type: "projection",
        },
      ],
    };
  }
  
  function getAttentionSubBlocks(arch) {
    const hd =
      arch.headDim ||
      (arch.hiddenSize && arch.numHeads
        ? Math.round(arch.hiddenSize / arch.numHeads)
        : undefined);
    return {
      layout: "threeWay",
      leftPath: [
        {
          name: "Q Projection",
          detail: arch.numHeads
            ? arch.numHeads + " heads" + (hd ? " \u00D7 " + hd + "d" : "")
            : "Query heads",
          type: "projection",
        },
        { name: "RoPE", detail: "Rotary Pos Emb", type: "operation" },
      ],
      middlePath: [
        {
          name: "K Projection",
          detail: arch.numKVHeads
            ? arch.numKVHeads +
              " KV heads" +
              (hd ? " \u00D7 " + hd + "d" : "") +
              " (shared)"
            : "Shared KV heads",
          type: "projection",
        },
        { name: "RoPE", detail: "Rotary Pos Emb", type: "operation" },
      ],
      rightPath: [
        {
          name: "V Projection",
          detail: arch.numKVHeads
            ? arch.numKVHeads +
              " KV heads" +
              (hd ? " \u00D7 " + hd + "d" : "")
            : "Value heads",
          type: "projection",
        },
      ],
      intermediateMergeBlocks: [],
      finalMergeBlocks: [
        {
          name: "Grouped Attention",
          detail:
            arch.numHeads && arch.numKVHeads
              ? arch.numHeads + ":" + arch.numKVHeads + " Q:KV ratio"
              : "Shared KV groups",
          type: "attention",
        },
        {
          name: "Output Projection",
          detail: arch.hiddenSize
            ? "\u2192 " + arch.hiddenSize.toLocaleString()
            : undefined,
          type: "projection",
        },
      ],
      leftLabel: "Q",
      middleLabel: "K",
      rightLabel: "V",
    };
  }
  
  // =========================================================================
  // Block color palette (light theme only — matches InferenceX light mode)
  // =========================================================================
  
  const BLOCK_COLORS = {
    embedding: { fill: "#dbeafe", stroke: "#3b82f6", expandBg: "#eff6ff" },
    attention: { fill: "#fef3c7", stroke: "#d97706", expandBg: "#fffbeb" },
    ffn: { fill: "#d1fae5", stroke: "#059669", expandBg: "#ecfdf5" },
    output: { fill: "#e0e7ff", stroke: "#6366f1", expandBg: "#eef2ff" },
    norm: { fill: "#f1f5f9", stroke: "#94a3b8", expandBg: "#f8fafc" },
    specs: { fill: "#f8fafc", stroke: "#cbd5e1", expandBg: "#f8fafc" },
    router: { fill: "#fce7f3", stroke: "#db2777", expandBg: "#fdf2f8" },
    expert: { fill: "#f3e8ff", stroke: "#9333ea", expandBg: "#faf5ff" },
    expertActive: { fill: "#e9d5ff", stroke: "#7c3aed", expandBg: "#f5f3ff" },
  };
  
  const SUB_BLOCK_COLORS = {
    projection: { fill: "#e0f2fe", stroke: "#0284c7" },
    activation: { fill: "#dcfce7", stroke: "#16a34a" },
    operation: { fill: "#f3f4f6", stroke: "#6b7280" },
    attention: { fill: "#fef9c3", stroke: "#ca8a04" },
  };
  

  // Exports
  A.MODEL_ARCHITECTURES = MODEL_ARCHITECTURES;
  A.formatParamCount = formatParamCount;
  A.formatContextWindow = formatContextWindow;
  A.getAttentionLabel = getAttentionLabel;
  A.escHtml = escHtml;
  A.getFFNSubBlocks = getFFNSubBlocks;
  A.getAttentionSubBlocks = getAttentionSubBlocks;
  A.computeLayerMemory = computeLayerMemory;
  A.BLOCK_COLORS = BLOCK_COLORS;
  A.SUB_BLOCK_COLORS = SUB_BLOCK_COLORS;
  A.fmtBytes = fmtBytes;
})();
