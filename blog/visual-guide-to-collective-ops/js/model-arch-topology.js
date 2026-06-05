/* Model Architectures — Device topology, collective operation diagrams.
 * Split from model-architectures.js for readability.
 * Depends on window.ArchLib (from model-arch-data.js + model-arch-memory.js).
 */
(function () {
  "use strict";
  var A = window.ArchLib = window.ArchLib || {};
  var escHtml = A.escHtml;
  var fmtGB = A.fmtGB;
  var computeMemory = A.computeMemory;
  var KV_DTYPES = A.KV_DTYPES;
  var GB = A.GB;

  // Device Topology — vertical GPU rows with toggleable layer details
  // =========================================================================
  
  var topoExpandedLayers = new Set();
  
  function renderDeviceTopology() {
    var panel = document.getElementById("arch-topo-panel");
    var state = A.getState ? A.getState() : {};
    var selectedArch = state.selectedArch;
    var memConfig = state.memConfig;
    if (!panel || !selectedArch || !memConfig) return;

    var arch = selectedArch;
    var cfg = memConfig;
    var m = computeMemory(arch, cfg);
    var isMoE = arch.architectureType === "moe";
    var tp = Math.max(1, Math.floor(cfg.tp));
    var dp = Math.max(1, Math.floor(cfg.dp));
    var totalGpus = dp * tp;
    var moeLayers = Math.max(0, (arch.numLayers || 0) - (isMoE ? (arch.denseFFNLayers || 0) : 0));
    var denseLayers2 = isMoE ? (arch.denseFFNLayers || 0) : 0;
    var routedCount = isMoE
      ? (arch.hasSharedExpert ? (arch.numExperts || 0) - 1 : arch.numExperts || 0)
      : 0;
    var numHeads = arch.numHeads || 0;
    var numKVHeads = arch.numKVHeads || numHeads;
    var epOn = isMoE && cfg.epEnabled && dp > 1;
    var ffnDim = arch.ffnDim || 0;
    var hiddenSize = arch.hiddenSize || 0;
  
    // --- Define layer blocks ---
    var headsPerTp = numHeads > 0 ? Math.floor(numHeads / tp) : 0;
    var kvHeadsPerTp = numKVHeads > 0 ? Math.floor(numKVHeads / Math.min(tp, numKVHeads)) : 0;
    var dimPerTp = ffnDim > 0 ? Math.floor(ffnDim / tp) : 0;
    var hidPerTp = hiddenSize > 0 ? Math.floor(hiddenSize / tp) : 0;
  
    var layerDefs = [];
    layerDefs.push({ key: "embed", label: "Embed", color: "#3b82f6", bg: "#dbeafe",
      desc: "Token \u2192 hidden (d=" + hiddenSize.toLocaleString() + ")" + (tp > 1 ? ". Column-parallel: each TP rank holds d/" + tp + " = " + hidPerTp + " columns." : ""),
      cellFn: function(d,t) {
        if (tp <= 1) return "full d=" + hiddenSize.toLocaleString();
        return "d[" + (t * hidPerTp) + "\u2013" + ((t+1) * hidPerTp - 1) + "]";
      },
      comm: tp > 1 ? { type: "allreduce", label: "AllReduce", color: "#3b82f6", desc: "sum partial embeddings across TP group" } : null
    });
    if (denseLayers2 > 0) {
      layerDefs.push({ key: "dense", label: "Dense\u00D7" + denseLayers2, color: "#059669", bg: "#d1fae5",
        desc: "Dense transformer (attn + FFN). " + (tp > 1 ? "Attn: " + headsPerTp + "/" + numHeads + " heads, FFN: gate+up col " + dimPerTp + "/" + (arch.denseFFNDim||ffnDim) + "." : ""),
        cellFn: function(d,t) { return tp > 1 ? headsPerTp + "h, ffn " + dimPerTp : numHeads + "h full"; },
        comm: tp > 1 ? { type: "allreduce", label: "AllReduce", color: "#3b82f6", desc: "after attn output + FFN down proj" } : null
      });
    }
    var attnLabelMap = {
      "AlternatingSinkGQA": "Alternating\nAttention (GQA)",
      "GQA": "Attention (GQA)",
      "MLA": "Attention (MLA)",
      "MHA": "Attention (MHA)"
    };
    var attnDisplayLabel = (attnLabelMap[arch.attentionType] || arch.attentionType || "Attention") + "\n\u00D7" + moeLayers + " layers";
    layerDefs.push({ key: "attn", label: attnDisplayLabel, color: "#d97706", bg: "#fef3c7",
      desc: numHeads + " Q heads, " + numKVHeads + " KV heads. " + (tp > 1 ? "Each TP rank: Q heads " + headsPerTp + "/" + numHeads + ", KV heads " + kvHeadsPerTp + "/" + numKVHeads + ". Q\u00B7K\u1d40\u00B7V computed independently per rank \u2014 no comm until output proj." : "All heads local."),
      cellFn: function(d,t) {
        if (tp <= 1) return "Q:" + numHeads + "h KV:" + numKVHeads + "h";
        return "Q h[" + (t * headsPerTp) + "\u2013" + ((t+1) * headsPerTp - 1) + "]\nKV " + kvHeadsPerTp + "/" + numKVHeads + "h";
      },
      comm: tp > 1 ? { type: "allreduce", label: "AllReduce", color: "#3b82f6", desc: "after output projection (within TP group)" } : null
    });
    if (isMoE) {
      layerDefs.push({ key: "moe", label: "MoE\u00D7" + moeLayers, color: "#dc2626", bg: "#fee2e2",
        desc: routedCount + " routed experts, top-" + (arch.activeExperts || "?") + (arch.hasSharedExpert ? " + 1 shared" : "") + ". " +
          (epOn ? "EP on: experts split across DP groups. All-to-All dispatch routes tokens to expert-owner GPU." : "EP off: all " + routedCount + " experts replicated on every GPU. Each GPU computes top-" + (arch.activeExperts || "?") + " locally.") +
          (tp > 1 ? " Each expert\u2019s SwiGLU weights column-sharded across TP." : ""),
        cellFn: function(d,t) {
          var ePerDp = epOn ? Math.ceil(routedCount / dp) : routedCount;
          var es = epOn ? d * ePerDp : 0;
          var ee = epOn ? Math.min(es + ePerDp - 1, routedCount - 1) : routedCount - 1;
          var range = "E" + es + "\u2013" + ee;
          if (tp > 1) range += "\nffn col[" + (t * dimPerTp) + "\u2013" + ((t+1) * dimPerTp - 1) + "]";
          return range;
        },
        bgFn: epOn ? function(d) {
          // Same red family, different opacity per DP group
          var opacities = [1.0, 0.6, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15];
          var op = opacities[d % opacities.length];
          return "rgba(254,202,202," + op + ")";
        } : null,
        commBefore: dp > 1 ? (epOn
          ? { type: "all2all", label: "All-to-All dispatch", color: "#9333ea", desc: "send each token to the GPU owning its expert" }
          : { type: "allgather", label: "AllGather", color: "#059669", desc: "replicate all tokens to every GPU" }
        ) : null,
        commAfter: dp > 1 ? (epOn
          ? { type: "all2all", label: "All-to-All combine", color: "#9333ea", desc: "return results to source GPU" }
          : { type: "reducescatter", label: "ReduceScatter", color: "#059669", desc: "sum partials, scatter back" }
        ) : null
      });
    } else {
      layerDefs.push({ key: "ffn", label: "FFN\u00D7" + moeLayers, color: "#059669", bg: "#d1fae5",
        desc: "SwiGLU. gate+up: col-parallel, down: row-parallel. dim=" + ffnDim.toLocaleString(),
        cellFn: function(d,t) {
          if (tp <= 1) return "dim=" + ffnDim.toLocaleString();
          return "gate+up col[" + (t*dimPerTp) + "\u2013" + ((t+1)*dimPerTp-1) + "]";
        },
        comm: tp > 1 ? { type: "allreduce", label: "AllReduce", color: "#3b82f6", desc: "after down projection (within TP group)" } : null
      });
    }
    layerDefs.push({ key: "lm", label: "LM Head", color: "#6366f1", bg: "#e0e7ff",
      desc: "vocab=" + (arch.vocabSize || "?").toLocaleString(),
      cellFn: function(d,t) {
        if (tp <= 1) return "full vocab";
        var vPerTp = Math.floor((arch.vocabSize || 0) / tp);
        return "vocab[" + (t*vPerTp).toLocaleString() + "\u2013" + ((t+1)*vPerTp-1).toLocaleString() + "]";
      }, comm: null
    });
    layerDefs.push({ key: "kv", label: "KV", color: "#0891b2", bg: "#cffafe",
      desc: m.perTokenKvKB.toFixed(1) + " KB/tok. " + (m.isMLA ? "MLA latent replicated across TP \u2014 raising TP does NOT increase KV." : "Sharded 1/" + m.kvShard + " across TP.") + " " + m.kvTokens.toLocaleString() + " tokens/replica.",
      cellFn: function(d,t) {
        return m.kvTokens.toLocaleString() + " tok\n" + (m.isMLA ? "MLA repl" : "KV h" + (t * kvHeadsPerTp) + "\u2013" + ((t+1)*kvHeadsPerTp - 1));
      }, comm: null
    });
  
    // --- Build HTML: TP groups as wrapper divs, internal grid per group ---
    var html = '<div class="arch-topo-header"><strong>Device Topology</strong>' +
      ' <span class="arch-badge-sm">' +
      (dp > 1 ? "DP" + dp + "\u00D7" : "") + (tp > 1 ? "TP" + tp + "\u00D7" : "") +
      totalGpus + "GPU" + (totalGpus > 1 ? "s" : "") + (epOn ? " EP" : "") + "</span></div>";
  
    html += '<div class="topo-outer">';
  
    for (var d0 = 0; d0 < dp; d0++) {
      // Gap between DP groups
      if (d0 > 0 && dp > 1) {
        html += '<div class="topo-dp-gap"></div>';
      }
  
      // TP group wrapper
      var tpGridCols = [];
      for (var t0 = 0; t0 < tp; t0++) {
        if (t0 > 0) tpGridCols.push("24px");
        tpGridCols.push("160px");
      }
  
      html += '<div class="topo-tp-box">';
      // Group label
      html += '<div class="topo-tp-label">';
      if (tp > 1) html += '<span class="topo-tp-title">TP Group</span> ';
      if (dp > 1) html += '<span class="topo-tp-dp">DP Replica ' + d0 + '</span>';
      if (tp > 1) html += '<span class="topo-tp-ar">\u21C4 AllReduce within</span>';
      html += '</div>';
  
      // Inner grid: GPU headers + layers
      html += '<div class="topo-tp-grid" style="grid-template-columns:' + tpGridCols.join(' ') + '">';
  
      // GPU headers
      for (var t0 = 0; t0 < tp; t0++) {
        if (t0 > 0) html += '<div class="topo-tp-sep-hdr"></div>';
        var gi = d0 * tp + t0;
        html += '<div class="topo-tp-gpu-hdr">GPU ' + gi;
        if (dp > 1) html += ' <span class="topo-v-tag">DP' + d0 + '</span>';
        if (tp > 1) html += '<span class="topo-v-tag">TP' + t0 + '</span>';
        html += '</div>';
      }
  
      // Layer rows
      for (var li = 0; li < layerDefs.length; li++) {
        var L = layerDefs[li];
        var isExp = topoExpandedLayers.has(L.key);
        for (var t0 = 0; t0 < tp; t0++) {
          if (t0 > 0) {
            // TP separator
            if (L.comm && L.comm.type === "allreduce") {
              html += '<div class="topo-tp-sep"><span style="color:#059669">\u21C4</span></div>';
            } else {
              html += '<div class="topo-tp-sep"></div>';
            }
          }
          var cellBg = (L.bgFn ? L.bgFn(d0) : L.bg);
          var cellText = L.cellFn(d0, t0);
          var moeCellAttr = (dp > 1 && isMoE && L.key === "moe") ? ' data-moe-cell="1"' : '';
          html += '<div class="topo-tp-layer' + (isExp ? ' expanded' : '') + '" data-topo-layer="' + L.key + '"' + moeCellAttr + ' style="background:' + cellBg + ';border-color:' + L.color + '" title="Click to expand">';
          html += '<div class="topo-h-layer-name" style="color:' + L.color + '">' + (isExp ? '\u25BE ' : '\u25B8 ') + escHtml(L.label).replace(/\n/g, '<br>') + '</div>';
          html += '<div class="topo-h-layer-val">' + escHtml(cellText).replace(/\n/g, '<br>') + '</div>';
          html += '</div>';
        }
      }
  
      html += '</div>'; // close topo-tp-grid
      html += '</div>'; // close topo-tp-box
    }


    html += '</div>';
  
    // --- Expanded layer detail (shown below the GPU stack) ---
    var anyExpanded = false;
    for (var li2 = 0; li2 < layerDefs.length; li2++) {
      var L2 = layerDefs[li2];
      if (!topoExpandedLayers.has(L2.key)) continue;
      anyExpanded = true;
      html += '<div class="topo-v-detail" style="border-left:3px solid ' + L2.color + '">';
      html += '<div class="topo-v-detail-title" style="color:' + L2.color + '">' + escHtml(L2.label) + '</div>';
  
      // Dataflow diagram + description + matched collective reference
      if (L2.key === "moe" && dp > 1) {
        html += '<div class="topo-v-detail-desc">' + escHtml(L2.desc) + '</div>';
        // Two-column: dataflow (left) + matched collective refs (right)
        html += '<div class="topo-detail-split">';
        html += '<div class="topo-detail-left">';
        html += buildMoEDataflowDiagram(arch, cfg, m, totalGpus, dp, tp, epOn, routedCount);
        html += '</div>';
        html += '<div class="topo-detail-right">';
        // Matched pairs with shaded background connecting them
        if (epOn) {
          html += '<div class="topo-coll-match" style="background:#9333ea08;border-color:#9333ea">';
          html += '<div class="topo-coll-match-header" style="color:#9333ea">Used in this layer: All-to-All</div>';
          html += buildCollectiveReference("all2all");
          html += '</div>';
        } else {
          html += '<div class="topo-coll-match" style="background:#0891b208;border-color:#0891b2">';
          html += '<div class="topo-coll-match-header" style="color:#0891b2">Used in this layer: AllGather + ReduceScatter</div>';
          html += buildCollectiveReference("allgather");
          html += buildCollectiveReference("reducescatter");
          html += '</div>';
        }
        html += '</div>';
        html += '</div>';
      } else if (L2.comm) {
        html += '<div class="topo-v-detail-desc">' + escHtml(L2.desc) + '</div>';
        html += '<div class="topo-detail-split">';
        html += '<div class="topo-detail-left">' + buildCollectiveDiagram(L2.comm, totalGpus, dp, tp) + '</div>';
        html += '<div class="topo-detail-right">';
        html += '<div class="topo-coll-match" style="background:' + L2.comm.color + '08;border-color:' + L2.comm.color + '">';
        html += '<div class="topo-coll-match-header" style="color:' + L2.comm.color + '">Used in this layer: ' + L2.comm.label + '</div>';
        html += buildCollectiveReference(L2.comm.type);
        html += '</div>';
        html += '</div>';
        html += '</div>';
      } else if (L2.key === "moe" && dp <= 1) {
        html += '<div class="topo-v-detail-desc">' + escHtml(L2.desc) + '</div>';
        html += '<div class="topo-v-diagram-wrap"><div class="topo-v-diagram-label" style="color:#16a34a">\u2714 Single replica \u2014 all ' + routedCount + ' experts local. Top-' + (arch.activeExperts || '?') + ' computed locally per token.</div></div>';
      } else if (L2.key === "kv") {
        html += '<div class="topo-v-detail-desc">' + escHtml(L2.desc) + '</div>';
      } else {
        html += '<div class="topo-v-detail-desc">' + escHtml(L2.desc) + '</div>';
      }
      html += '</div>'; // close detail
    }
  
    // --- Collective Operations Reference (toggle) ---
    html += '<div class="topo-coll-ref">';
    html += '<button type="button" class="topo-coll-ref-toggle" onclick="this.parentElement.classList.toggle(\'open\')">';
    html += '<span class="topo-coll-ref-icon">\u25B8</span> Collective Operations Reference</button>';
    html += '<div class="topo-coll-ref-body">';
    html += buildCollectiveReference("allreduce");
    html += buildCollectiveReference("allgather");
    html += buildCollectiveReference("reducescatter");
    html += buildCollectiveReference("all2all");
    html += buildCollectiveReference("broadcast");
    html += buildCollectiveReference("reduce");
    html += '</div></div>';
  
    // Footer
    if (isMoE && dp > 1) {
      html += '<div class="topo-v-footer">' + (epOn
        ? "EP on: routed tokens cross point-to-point (\u00D7" + (moeLayers*2) + "/fwd)"
        : "EP off: experts replicated, AG+RS per MoE layer (\u00D7" + (moeLayers*2) + "/fwd)") + '</div>';
    }
  
    // Hint
    if (!anyExpanded) {
      html += '<div class="topo-v-footer" style="color:#9ca3af;font-style:italic">\u25B8 Click any layer block above to see how it\u2019s sharded and which collective operations are used</div>';
    }
  
    panel.innerHTML = html;

    // Draw MoE bounding box overlay (EP-on or EP-off, dp>1)
    if (isMoE && dp > 1) {
      var bboxLabel = epOn ? "All-to-All" : "AllGather + ReduceScatter";
      var bboxSub = epOn
        ? "each token dispatched to expert-owner GPU (\u00D7" + (moeLayers * 2) + "/fwd)"
        : "tokens replicated \u2192 local compute \u2192 scatter (\u00D7" + (moeLayers * 2) + "/fwd)";
      var bboxColor = epOn ? "#dc2626" : "#0891b2";
      drawMoEBoundingBox(panel, bboxLabel, bboxSub, bboxColor);
      if (A._moeBoxResizeHandler) window.removeEventListener("resize", A._moeBoxResizeHandler);
      A._moeBoxResizeHandler = function () { drawMoEBoundingBox(panel, bboxLabel, bboxSub, bboxColor); };
      window.addEventListener("resize", A._moeBoxResizeHandler);
    }

    // Wire up toggle clicks
    panel.querySelectorAll('[data-topo-layer]').forEach(function(el) {
      el.addEventListener('click', function(e) {
        e.stopPropagation();
        var key = el.dataset.topoLayer;
        if (topoExpandedLayers.has(key)) topoExpandedLayers.delete(key);
        else topoExpandedLayers.add(key);
        renderDeviceTopology();
      });
    });
  }
  
  
  
  // Dashed bounding box overlay around the MoE row (EP-on or EP-off, dp>1).
  // Grid layout makes a spanning box impossible in pure CSS, so measure after render.
  function drawMoEBoundingBox(panel, labelText, subText, color) {
    // Remove any previous overlay (renderDeviceTopology re-runs on toggle clicks)
    panel.querySelectorAll(".topo-moe-bbox, .topo-moe-bbox-label").forEach(function (el) { el.remove(); });

    var cells = panel.querySelectorAll('[data-moe-cell]');
    if (!cells.length) return;

    var pcs = window.getComputedStyle(panel);
    if (pcs.position === "static") panel.style.position = "relative";

    var pr = panel.getBoundingClientRect();
    var minL = Infinity, minT = Infinity, maxR = -Infinity, maxB = -Infinity;
    cells.forEach(function (c) {
      var r = c.getBoundingClientRect();
      minL = Math.min(minL, r.left);
      minT = Math.min(minT, r.top);
      maxR = Math.max(maxR, r.right);
      maxB = Math.max(maxB, r.bottom);
    });

    var pad = 8;
    var left = minL - pr.left + panel.scrollLeft - pad;
    var top = minT - pr.top + panel.scrollTop - pad;
    var width = (maxR - minL) + pad * 2;
    var height = (maxB - minT) + pad * 2;

    var box = document.createElement("div");
    box.className = "topo-moe-bbox";
    box.style.left = left + "px";
    box.style.top = top + "px";
    box.style.width = width + "px";
    box.style.height = height + "px";
    box.style.borderColor = color;
    box.style.background = color + "08";
    panel.appendChild(box);

    var label = document.createElement("div");
    label.className = "topo-moe-bbox-label";
    label.innerHTML = '<span class="topo-moe-bbox-name">' + escHtml(labelText) + '</span>'
      + '<span class="topo-moe-bbox-sub">' + escHtml(subText) + '</span>';
    label.style.left = (left + width + 12) + "px";
    label.style.top = (top + height / 2) + "px";
    label.style.color = color;
    panel.appendChild(label);
  }

  // Generic collective diagram (AllReduce) — GPUs horizontal, data top→bottom
  // Grayscale 1×4 vector grid — ranks stacked vertically, bounding boxes
  var GFILL = {0:"#fff",1:"#e5e7eb",3:"#9ca3af",5:"#6b7280",7:"#374151",9:"#111827"};
  var GTXT  = {0:"#d1d5db",1:"#6b7280",3:"#374151",5:"#f3f4f6",7:"#f3f4f6",9:"#f3f4f6"};
  var RBOX  = ["#3b82f6","#f59e0b"];

  function buildCollectiveGrid(d) {
    if (!d.grid) return "";
    var cols = d.grid.cols || 4, cell = 24, gp = 3, bpad = 6;
    var gW = cols*cell + (cols-1)*gp;
    var bW = gW + bpad*2;
    var labelH = 16;
    var bH = cell + bpad*2 + labelH;
    var pad = 10, arW = 30, gap = 30;
    var id = d.name.replace(/[^a-z]/gi,"");

    // Fixed y-positions: R0 and R1 at same y across ALL 3 columns
    var r0y = pad;
    var r1y = pad + bH + gap;
    var totalH = 2*bH + gap;

    var nA = d.grid.after.length;
    var svgW = pad + bW + arW + bW + arW + bW + pad;
    var legendH = 22;
    var svgH = pad + totalH + 12 + legendH + pad;

    // renderGrid: lblBot=true → cells at top, label at bottom
    function renderGrid(ox, oy, data, label, boxColor, lblBot) {
      var o = '';
      o += '<rect x="'+ox+'" y="'+oy+'" width="'+bW+'" height="'+bH+'" rx="6" fill="none" stroke="#d1d5db" stroke-width="1" stroke-dasharray="4,2"/>';
      var cellY = lblBot ? oy + bpad : oy + labelH + bpad;
      var lblY = lblBot ? oy + bH - 4 : oy + 12;
      o += '<text x="'+(ox+bW/2)+'" y="'+lblY+'" text-anchor="middle" font-size="11" font-weight="700" fill="'+boxColor+'">'+label+'</text>';
      for (var c=0; c<cols; c++) {
        var val = data[c];
        var cx = ox + bpad + c*(cell+gp);
        o += '<rect x="'+cx+'" y="'+cellY+'" width="'+cell+'" height="'+cell+'" rx="3" fill="'+(GFILL[val]||GFILL[0])+'" stroke="#e5e7eb" stroke-width="0.5"/>';
        if (val === 0) {
          o += '<text x="'+(cx+cell/2)+'" y="'+(cellY+cell/2+1)+'" text-anchor="middle" dominant-baseline="central" font-size="11" fill="#d1d5db">\u2205</text>';
        } else {
          o += '<text x="'+(cx+cell/2)+'" y="'+(cellY+cell/2+1)+'" text-anchor="middle" dominant-baseline="central" font-size="11" font-weight="700" fill="'+(GTXT[val]||"#374151")+'">'+val+'</text>';
        }
      }
      return o;
    }

    var svg = '<svg width="'+svgW+'" height="'+svgH+'" style="font-family:\'Montserrat\',sans-serif;display:block;margin:0 auto">';
    svg += '<defs><marker id="gc-'+id+'" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="'+d.color+'" stroke-width="1.5" stroke-linecap="round"/></marker></defs>';

    var r0cy = r0y + bH/2, r1cy = r1y + bH/2;

    // --- Column 1: Before ---
    var col1 = pad;
    svg += renderGrid(col1, r0y, d.grid.before[0], "Rank 0", RBOX[0]);
    svg += renderGrid(col1, r1y, d.grid.before[1], "Rank 1", RBOX[1], true);

    // Arrows 1: per-rank, Before → Operation
    var col2 = col1 + bW + arW;
    svg += '<line x1="'+(col1+bW+4)+'" y1="'+r0cy+'" x2="'+(col2-4)+'" y2="'+r0cy+'" stroke="'+d.color+'" stroke-width="1" marker-end="url(#gc-'+id+')"/>';
    svg += '<line x1="'+(col1+bW+4)+'" y1="'+r1cy+'" x2="'+(col2-4)+'" y2="'+r1cy+'" stroke="'+d.color+'" stroke-width="1" marker-end="url(#gc-'+id+')"/>';

    // --- Column 2: Operation ---
    svg += renderGrid(col2, r0y, d.grid.before[0], "Rank 0", RBOX[0]);
    svg += renderGrid(col2, r1y, d.grid.before[1], "Rank 1", RBOX[1], true);

    // Connection lines between R0 cells (bottom) and R1 cells (top)
    if (d.grid.lines) {
      var r0bot = r0y + labelH + bpad + cell;
      var r1top = r1y + bpad;
      var linesMid = (r0bot + r1top) / 2;
      var lines = d.grid.lines;
      for (var li=0; li<lines.length; li++) {
        var fromC = lines[li][0], toC = lines[li][1], dir = lines[li][2] || "both";
        var lx1 = col2 + bpad + fromC*(cell+gp) + cell/2;
        var lx2 = col2 + bpad + toC*(cell+gp) + cell/2;
        var isCross = fromC !== toC;
        var lOp = isCross ? 1 : 0.35;
        svg += '<line x1="'+lx1+'" y1="'+r0bot+'" x2="'+lx2+'" y2="'+r1top+'" stroke="'+d.color+'" stroke-width="'+(isCross ? '1.2' : '0.8')+'" opacity="'+lOp+'" stroke-dasharray="'+(isCross ? '4,2' : '3,2')+'"/>';
        // Arrow triangles aligned to line direction
        var aSz = 3.5, aH = aSz*1.8;
        var ldx = lx2-lx1, ldy = r1top-r0bot;
        var lLen = Math.sqrt(ldx*ldx+ldy*ldy)||1;
        var cs = ldx/lLen, sn = ldy/lLen;
        if (dir === "up" || dir === "both") {
          // tip at R0 end, pointing away from R1
          var bx1=lx1+aH*cs, by1=r0bot+aH*sn;
          svg += '<polygon points="'+lx1+','+r0bot+' '+(bx1+aSz*sn)+','+(by1-aSz*cs)+' '+(bx1-aSz*sn)+','+(by1+aSz*cs)+'" fill="'+d.color+'" opacity="'+lOp+'"/>';
        }
        if (dir === "down" || dir === "both") {
          // tip at R1 end, pointing away from R0
          var bx2=lx2-aH*cs, by2=r1top-aH*sn;
          svg += '<polygon points="'+lx2+','+r1top+' '+(bx2+aSz*sn)+','+(by2-aSz*cs)+' '+(bx2-aSz*sn)+','+(by2+aSz*cs)+'" fill="'+d.color+'" opacity="'+lOp+'"/>';
        }
        if (d.grid.op) {
          var v0 = d.grid.before[0][fromC], v1 = d.grid.before[1][toC];
          var vr = Math.max(v0, v1);
          var mx = (lx1+lx2)/2;
          svg += '<rect x="'+(mx-11)+'" y="'+(linesMid-7)+'" width="22" height="14" rx="3" fill="#fff" stroke="#e5e7eb" stroke-width="0.4"/>';
          svg += '<text x="'+mx+'" y="'+(linesMid+1)+'" text-anchor="middle" dominant-baseline="central" font-size="8" font-weight="700" fill="'+d.color+'">'+vr+'</text>';
        }
      }
      // Op label: left of the connection lines
      if (d.grid.opLabel) {
        svg += '<text x="'+(col2)+'" y="'+(linesMid+1)+'" text-anchor="end" dominant-baseline="central" font-size="10" font-weight="700" fill="'+d.color+'">'+d.grid.opLabel+'</text>';
      }
    }

    // Arrows 2: Operation → Result
    var col3 = col2 + bW + arW;
    if (nA === 1) {
      // Single result: Y-merge — both rank arrows converge
      var singleY = pad + (totalH - bH) / 2;
      var singleCy = singleY + bH / 2;
      var mergeX = col2 + bW + arW * 0.55;
      // R0 curve down to merge
      svg += '<path d="M'+(col2+bW+4)+','+r0cy+' Q'+mergeX+','+r0cy+' '+mergeX+','+singleCy+'" fill="none" stroke="'+d.color+'" stroke-width="1"/>';
      // R1 curve up to merge
      svg += '<path d="M'+(col2+bW+4)+','+r1cy+' Q'+mergeX+','+r1cy+' '+mergeX+','+singleCy+'" fill="none" stroke="'+d.color+'" stroke-width="1"/>';
      // Final arrow to result
      svg += '<line x1="'+mergeX+'" y1="'+singleCy+'" x2="'+(col3-4)+'" y2="'+singleCy+'" stroke="'+d.color+'" stroke-width="1" marker-end="url(#gc-'+id+')"/>';
      svg += renderGrid(col3, singleY, d.grid.after[0], "All Ranks", d.color);
    } else {
      // Dual result: per-rank arrows
      svg += '<line x1="'+(col2+bW+4)+'" y1="'+r0cy+'" x2="'+(col3-4)+'" y2="'+r0cy+'" stroke="'+d.color+'" stroke-width="1" marker-end="url(#gc-'+id+')"/>';
      svg += '<line x1="'+(col2+bW+4)+'" y1="'+r1cy+'" x2="'+(col3-4)+'" y2="'+r1cy+'" stroke="'+d.color+'" stroke-width="1" marker-end="url(#gc-'+id+')"/>';
      svg += renderGrid(col3, r0y, d.grid.after[0], "Rank 0", RBOX[0]);
      svg += renderGrid(col3, r1y, d.grid.after[1], "Rank 1", RBOX[1], true);
    }

    // Legend
    var ly = pad + totalH + 12;
    var vals = [1,3,5,7,9];
    for (var vi=0; vi<vals.length; vi++) {
      var vx = pad+vi*34, v = vals[vi];
      svg += '<rect x="'+vx+'" y="'+ly+'" width="14" height="14" rx="2" fill="'+GFILL[v]+'" stroke="#cbd5e1" stroke-width="0.4"/>';
      svg += '<text x="'+(vx+18)+'" y="'+(ly+10)+'" font-size="9" fill="#6b7280">='+v+'</text>';
    }
    var ex = pad+5*34+6;
    svg += '<rect x="'+ex+'" y="'+ly+'" width="14" height="14" rx="2" fill="#fff" stroke="#cbd5e1" stroke-width="0.4"/>';
    svg += '<text x="'+(ex+18)+'" y="'+(ly+10)+'" font-size="9" fill="#9ca3af">\u2205</text>';

    svg += '</svg>';
    return svg;
  }

  // NCCL-style rank-row collective diagrams: colored segments show data origin
  var RCOL = ["#3b82f6","#f59e0b"];

  function buildCollectiveReference(type) {
    var defs = {
      allreduce: {
        name: "AllReduce", color: "#e11d48",
        desc: "Each rank has a partial result. The op is applied element-wise across all ranks, and every rank receives the same final result.",
        seg: { before: [[0,0],[1,1]], after: [[-2,-2],[-2,-2]] },
        grid: { op:"max", opLabel:"operation: max", cols:4, before: [[9,3,5,1],[3,7,1,5]],
          lines: [[0,0,"both"],[1,1,"both"],[2,2,"both"],[3,3,"both"]], after: [[9,7,5,5]] }
      },
      allgather: {
        name: "AllGather", color: "#e11d48",
        desc: "Each rank has its own chunk. After AllGather, every rank has the concatenation of all chunks.",
        seg: { before: [[0,-1],[-1,1]], after: [[0,1],[0,1]] },
        grid: { opLabel:"concat", cols:4, before: [[9,3,0,0],[0,0,5,7]],
          lines: [[0,0,"down"],[1,1,"down"],[2,2,"up"],[3,3,"up"]], after: [[9,3,5,7]] }
      },
      reducescatter: {
        name: "ReduceScatter", color: "#e11d48",
        desc: "Each rank has the full data. The op is applied element-wise, and each rank receives only its own reduced portion.",
        seg: { before: [[0,0],[1,1]], after: [[-2,-1],[-1,-2]] },
        grid: { op:"max", opLabel:"operation: max", cols:4, before: [[9,3,5,1],[3,7,1,5]],
          lines: [[0,0,"up"],[1,1,"up"],[2,2,"down"],[3,3,"down"]], after: [[9,7,0,0],[0,0,5,5]] }
      },
      all2all: {
        name: "All-to-All", color: "#e11d48",
        desc: "Each rank sends different data to each other rank. Rank i\u2019s j-th chunk goes to rank j. Point-to-point.",
        seg: { before: [[0,0],[1,1]], after: [[0,1],[0,1]] },
        grid: { opLabel:"exchange", cols:4, before: [[9,3,5,1],[7,1,3,5]],
          lines: [[2,0,"both"],[3,1,"both"]], after: [[9,3,7,1],[5,1,3,5]] }
      },
      broadcast: {
        name: "Broadcast", color: "#e11d48",
        desc: "One rank (src) sends its data to all other ranks. After broadcast, every rank has the same data as the source.",
        seg: { before: [[0,0],[-1,-1]], after: [[0,0],[0,0]] },
        grid: { opLabel:"copy", cols:4, before: [[9,3,5,7],[0,0,0,0]],
          lines: [[0,0,"down"],[1,1,"down"],[2,2,"down"],[3,3,"down"]], after: [[9,3,5,7]] }
      },
      reduce: {
        name: "Reduce", color: "#e11d48",
        desc: "Each rank has a partial result. The op is applied element-wise, but only the destination rank receives the final result.",
        seg: { before: [[0,0],[1,1]], after: [[-2,-2],[-1,-1]] },
        grid: { op:"max", opLabel:"operation: max", cols:4, before: [[9,3,5,1],[3,7,1,5]],
          lines: [[0,0,"up"],[1,1,"up"],[2,2,"up"],[3,3,"up"]], after: [[9,7,5,5],[0,0,0,0]] }
      }
    };

    var d = defs[type];
    if (!d) return "";

    // Segment diagram dimensions
    var nR = 2, nSeg = d.seg.before[0].length;
    var segW = 42, segH = 24, segGap = 2;
    var labelW = 46, hdrH = 16, rowGap = 6, arrowW = 76, pad = 10;
    var secW = nSeg*segW + (nSeg-1)*segGap;
    var bodyH = nR*segH + (nR-1)*rowGap;
    var svgW = pad + labelW + secW + arrowW + labelW + secW + pad;
    var svgH = pad + hdrH + bodyH + pad;

    var s = '<div class="topo-coll-card" style="border-color:' + d.color + '20">';
    s += '<div class="topo-coll-card-name" style="color:' + d.color + '">' + d.name + '</div>';
    s += '<div class="topo-coll-card-desc">' + escHtml(d.desc) + '</div>';
    s += '<div class="topo-coll-card-viz"><div class="topo-coll-viz-left">';
    s += '<svg width="'+svgW+'" height="'+svgH+'" style="font-family:\'Montserrat\',sans-serif;display:block;margin:0 auto">';

    // Defs: stripe pattern for reduced, arrow marker
    s += '<defs>';
    s += '<pattern id="rs-'+type+'" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">';
    s += '<rect width="3" height="6" fill="'+RCOL[0]+'"/><rect x="3" width="3" height="6" fill="'+RCOL[1]+'"/>';
    s += '</pattern>';
    s += '<marker id="ca-'+type+'" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="'+d.color+'" stroke-width="1.5" stroke-linecap="round"/></marker>';
    s += '</defs>';

    // Column headers
    var bx = pad+labelW, ax = bx+secW+arrowW+labelW;
    s += '<text x="'+(bx+secW/2)+'" y="'+(pad+11)+'" text-anchor="middle" font-size="9" font-weight="600" fill="#9ca3af">Before</text>';
    s += '<text x="'+(ax+secW/2)+'" y="'+(pad+11)+'" text-anchor="middle" font-size="9" font-weight="600" fill="#9ca3af">After</text>';

    function renderSeg(sx, sy, val) {
      var fill, txt, tc;
      if (val===-1) { fill="#f3f4f6"; txt="\u2205"; tc="#d1d5db"; }
      else if (val===-2) { fill="url(#rs-"+type+")"; txt="\u03A3"; tc="#fff"; }
      else { fill=RCOL[val]; txt=["A","B"][val]; tc="#fff"; }
      return '<rect x="'+sx+'" y="'+sy+'" width="'+segW+'" height="'+segH+'" rx="4" fill="'+fill+'" stroke="#e5e7eb" stroke-width="0.5"/>'
        + '<text x="'+(sx+segW/2)+'" y="'+(sy+segH/2+1)+'" text-anchor="middle" dominant-baseline="central" font-size="10" font-weight="600" fill="'+tc+'">'+txt+'</text>';
    }

    for (var ri=0; ri<nR; ri++) {
      var ry = pad+hdrH+ri*(segH+rowGap);
      // Before rank label + segments
      s += '<text x="'+(pad+labelW-6)+'" y="'+(ry+segH/2+1)+'" text-anchor="end" dominant-baseline="central" font-size="9" font-weight="600" fill="'+RCOL[ri]+'">Rank '+ri+'</text>';
      for (var si=0; si<nSeg; si++) {
        s += renderSeg(bx+si*(segW+segGap), ry, d.seg.before[ri][si]);
      }
      // After rank label + segments
      s += '<text x="'+(ax-6)+'" y="'+(ry+segH/2+1)+'" text-anchor="end" dominant-baseline="central" font-size="9" font-weight="600" fill="'+RCOL[ri]+'">Rank '+ri+'</text>';
      for (var si2=0; si2<nSeg; si2++) {
        s += renderSeg(ax+si2*(segW+segGap), ry, d.seg.after[ri][si2]);
      }
    }

    // Arrow between sections
    var ay = pad+hdrH+bodyH/2;
    s += '<line x1="'+(bx+secW+6)+'" y1="'+ay+'" x2="'+(ax-labelW-6)+'" y2="'+ay+'" stroke="'+d.color+'" stroke-width="1.2" marker-end="url(#ca-'+type+')"/>';
    s += '<text x="'+((bx+secW+ax-labelW)/2)+'" y="'+(ay-7)+'" text-anchor="middle" font-size="8" font-weight="600" fill="'+d.color+'">'+d.name+'</text>';

    s += '</svg>';
    s += '</div>';
    s += '<div class="topo-coll-viz-right">' + buildCollectiveGrid(d) + '</div></div>';
    s += '</div>';
    return s;
  }
  
  function buildCollectiveDiagram(comm, totalGpus, dp, tp) {
    var n = Math.min(totalGpus, 6);
    var boxW = 90, boxH = 36, hGap = 18, pad2 = 16;
    var svgW = n * boxW + (n - 1) * hGap + pad2 * 2;
    var arrowH = 18, bandH = 22;
    var svgH = boxH + arrowH + bandH + arrowH + boxH + pad2 * 2;
  
    function gpuCx(i) { return pad2 + i * (boxW + hGap) + boxW / 2; }
    function gpuLx(i) { return pad2 + i * (boxW + hGap); }
  
    var s = '<div class="topo-v-diagram-wrap">';
    s += '<div class="topo-v-diagram-label" style="color:' + comm.color + '">' + escHtml(comm.label) + '</div>';
    s += '<svg width="' + svgW + '" height="' + svgH + '" style="font-family:\'Montserrat\',sans-serif;overflow:visible">';
    s += '<defs><marker id="tc-ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="' + comm.color + '" stroke-width="1.5" stroke-linecap="round"/></marker></defs>';
  
    var y = pad2;
    // Before
    for (var i = 0; i < n; i++) {
      s += '<rect x="' + gpuLx(i) + '" y="' + y + '" width="' + boxW + '" height="' + boxH + '" rx="4" fill="#f3f4f6" stroke="#9ca3af" stroke-width="0.6"/>';
      s += '<text x="' + gpuCx(i) + '" y="' + (y + 11) + '" text-anchor="middle" font-size="12" font-weight="600" fill="#374151">GPU ' + i + '</text>';
      s += '<text x="' + gpuCx(i) + '" y="' + (y + 22) + '" text-anchor="middle" font-size="11" fill="#6b7280">partial</text>';
    }
    y += boxH;
    for (var i2 = 0; i2 < n; i2++) s += '<line x1="' + gpuCx(i2) + '" y1="' + y + '" x2="' + gpuCx(i2) + '" y2="' + (y + arrowH - 3) + '" stroke="#9ca3af" stroke-width="1" marker-end="url(#tc-ar)"/>';
    y += arrowH;
    // Band
    s += '<rect x="' + pad2 + '" y="' + y + '" width="' + (svgW - pad2*2) + '" height="' + bandH + '" rx="3" fill="' + comm.color + '" opacity="0.1"/>';
    s += '<text x="' + (svgW/2) + '" y="' + (y + bandH/2 + 1) + '" text-anchor="middle" font-size="11" font-weight="600" fill="' + comm.color + '">' + escHtml(comm.label) + ': ' + escHtml(comm.desc) + '</text>';
    for (var i3 = 0; i3 < n-1; i3++) s += '<line x1="' + (gpuLx(i3)+boxW) + '" y1="' + (y+bandH/2) + '" x2="' + gpuLx(i3+1) + '" y2="' + (y+bandH/2) + '" stroke="' + comm.color + '" stroke-width="1" stroke-dasharray="3,2" opacity="0.4"/>';
    y += bandH;
    for (var i4 = 0; i4 < n; i4++) s += '<line x1="' + gpuCx(i4) + '" y1="' + y + '" x2="' + gpuCx(i4) + '" y2="' + (y + arrowH - 3) + '" stroke="#9ca3af" stroke-width="1" marker-end="url(#tc-ar)"/>';
    y += arrowH;
    // After
    for (var i5 = 0; i5 < n; i5++) {
      s += '<rect x="' + gpuLx(i5) + '" y="' + y + '" width="' + boxW + '" height="' + boxH + '" rx="4" fill="' + comm.color + '15" stroke="' + comm.color + '" stroke-width="0.6"/>';
      s += '<text x="' + gpuCx(i5) + '" y="' + (y + boxH/2 + 1) + '" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="600" fill="' + comm.color + '">same result</text>';
    }
    s += '</svg></div>';
    return s;
  }
  
  // Full MoE dataflow diagram: shows the complete EP off vs EP on flow
  function buildMoEDataflowDiagram(arch, cfg, m, totalGpus, dp, tp, epOn, routedCount) {
    var n = Math.min(totalGpus, 6);
    var boxW = 100, boxH = 42, hGap = 20, pad2 = 16;
    var svgW = n * boxW + (n - 1) * hGap + pad2 * 2;
    var arrowH = 20, bandH = 26;
    var svgH = boxH + arrowH + bandH + arrowH + boxH + arrowH + bandH + arrowH + boxH + pad2 * 2 + 16;
  
    function gpuCx(i) { return pad2 + i * (boxW + hGap) + boxW / 2; }
    function gpuLx(i) { return pad2 + i * (boxW + hGap); }
  
    var bandColor = epOn ? "#9333ea" : "#059669";
    var bandLabel1 = epOn ? "All-to-All dispatch" : "AllGather";
    var bandDesc1 = epOn ? "send each token \u2192 expert-owner GPU" : "every GPU gets ALL GPUs\u2019 tokens";
    var bandLabel2 = epOn ? "All-to-All combine" : "ReduceScatter";
    var bandDesc2 = epOn ? "return results \u2192 source GPU" : "sum partials, scatter back";
  
    var s = '<div class="topo-v-diagram-wrap">';
    s += '<div class="topo-v-diagram-label" style="color:' + bandColor + '">' + (epOn ? 'EP on' : 'EP off') + ' \u2014 MoE dataflow per layer</div>';
    s += '<svg width="' + svgW + '" height="' + svgH + '" style="font-family:\'Montserrat\',sans-serif;overflow:visible">';
    s += '<defs><marker id="tc-moe" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round"/></marker></defs>';
  
    var y = pad2;
  
    // Row 1: tokens / router per GPU
    for (var i = 0; i < n; i++) {
      s += '<rect x="' + gpuLx(i) + '" y="' + y + '" width="' + boxW + '" height="' + boxH + '" rx="5" fill="#f1f5f9" stroke="#94a3b8" stroke-width="0.6"/>';
      s += '<text x="' + gpuCx(i) + '" y="' + (y + 13) + '" text-anchor="middle" font-size="12" font-weight="600" fill="#374151">GPU ' + i + '</text>';
      s += '<text x="' + gpuCx(i) + '" y="' + (y + 26) + '" text-anchor="middle" font-size="11" fill="#6b7280">tokens + router</text>';
    }
    y += boxH;
    for (var i2 = 0; i2 < n; i2++) s += '<line x1="' + gpuCx(i2) + '" y1="' + y + '" x2="' + gpuCx(i2) + '" y2="' + (y+arrowH-4) + '" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#tc-moe)"/>';
    y += arrowH;
  
    // Band 1: dispatch / allgather
    s += '<rect x="' + pad2 + '" y="' + y + '" width="' + (svgW-pad2*2) + '" height="' + bandH + '" rx="4" fill="' + bandColor + '" opacity="0.1"/>';
    s += '<text x="' + (svgW/2) + '" y="' + (y + 11) + '" text-anchor="middle" font-size="11" font-weight="600" fill="' + bandColor + '">' + bandLabel1 + '</text>';
    s += '<text x="' + (svgW/2) + '" y="' + (y + 23) + '" text-anchor="middle" font-size="11" fill="' + bandColor + '" opacity="0.7">' + bandDesc1 + '</text>';
    // Connection: dashed for AG, curved arrows for A2A
    if (epOn) {
      for (var i3 = 0; i3 < n; i3++) {
        for (var j = 0; j < n; j++) {
          if (i3 === j) continue;
          if (Math.abs(i3 - j) > 2) continue; // keep diagram readable
          var fx = gpuCx(i3), tx = gpuCx(j);
          s += '<path d="M' + fx + ' ' + (y+bandH) + ' Q' + ((fx+tx)/2) + ' ' + (y+bandH+10) + ' ' + tx + ' ' + (y+bandH) + '" fill="none" stroke="' + bandColor + '" stroke-width="0.8" opacity="0.3" marker-end="url(#tc-moe)"/>';
        }
      }
    } else {
      for (var i4 = 0; i4 < n-1; i4++) s += '<line x1="' + (gpuLx(i4)+boxW) + '" y1="' + (y+bandH/2) + '" x2="' + gpuLx(i4+1) + '" y2="' + (y+bandH/2) + '" stroke="' + bandColor + '" stroke-width="1" stroke-dasharray="3,2" opacity="0.5"/>';
    }
    y += bandH;
    for (var i5 = 0; i5 < n; i5++) s += '<line x1="' + gpuCx(i5) + '" y1="' + y + '" x2="' + gpuCx(i5) + '" y2="' + (y+arrowH-4) + '" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#tc-moe)"/>';
    y += arrowH;
  
    // Row 2: expert compute boxes
    var ePerGpu = epOn ? Math.ceil(routedCount / n) : routedCount;
    var expertHues = ["#f3e8ff","#fce7f3","#e0f2fe","#ecfdf5","#fef3c7","#eef2ff"];
    for (var i6 = 0; i6 < n; i6++) {
      var es, ee;
      if (epOn) { es = i6 * ePerGpu; ee = Math.min(es + ePerGpu - 1, routedCount - 1); }
      else { es = 0; ee = routedCount - 1; }
      var eFill = epOn ? expertHues[i6 % expertHues.length] : "#e6f1fb";
      s += '<rect x="' + gpuLx(i6) + '" y="' + y + '" width="' + boxW + '" height="' + boxH + '" rx="5" fill="' + eFill + '" stroke="#9333ea" stroke-width="0.6"/>';
      s += '<text x="' + gpuCx(i6) + '" y="' + (y + 13) + '" text-anchor="middle" font-size="11" font-weight="600" fill="#581c87">E' + es + '\u2013E' + ee + '</text>';
      s += '<text x="' + gpuCx(i6) + '" y="' + (y + 27) + '" text-anchor="middle" font-size="11" fill="#7c3aed">' + (epOn ? 'local compute' : 'compute locally') + '</text>';
    }
    // Note under expert boxes
    if (!epOn) {
      s += '<text x="' + (svgW/2) + '" y="' + (y + boxH + 12) + '" text-anchor="middle" font-size="11" fill="#6b7280">each GPU holds all experts \u2014 no expert is remote</text>';
    }
    y += boxH;
    for (var i7 = 0; i7 < n; i7++) s += '<line x1="' + gpuCx(i7) + '" y1="' + (y + (epOn ? 0 : 14)) + '" x2="' + gpuCx(i7) + '" y2="' + (y+arrowH-4 + (epOn ? 0 : 14)) + '" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#tc-moe)"/>';
    y += arrowH + (epOn ? 0 : 14);
  
    // Band 2: combine / reducescatter
    s += '<rect x="' + pad2 + '" y="' + y + '" width="' + (svgW-pad2*2) + '" height="' + bandH + '" rx="4" fill="' + bandColor + '" opacity="0.1"/>';
    s += '<text x="' + (svgW/2) + '" y="' + (y + 11) + '" text-anchor="middle" font-size="11" font-weight="600" fill="' + bandColor + '">' + bandLabel2 + '</text>';
    s += '<text x="' + (svgW/2) + '" y="' + (y + 23) + '" text-anchor="middle" font-size="11" fill="' + bandColor + '" opacity="0.7">' + bandDesc2 + '</text>';
    if (epOn) {
      for (var i8 = 0; i8 < n; i8++) {
        for (var j2 = 0; j2 < n; j2++) {
          if (i8 === j2) continue;
          if (Math.abs(i8 - j2) > 2) continue;
          var fx2 = gpuCx(i8), tx2 = gpuCx(j2);
          s += '<path d="M' + fx2 + ' ' + (y+bandH) + ' Q' + ((fx2+tx2)/2) + ' ' + (y+bandH+10) + ' ' + tx2 + ' ' + (y+bandH) + '" fill="none" stroke="' + bandColor + '" stroke-width="0.8" opacity="0.3" marker-end="url(#tc-moe)"/>';
        }
      }
    } else {
      for (var i9 = 0; i9 < n-1; i9++) s += '<line x1="' + (gpuLx(i9)+boxW) + '" y1="' + (y+bandH/2) + '" x2="' + gpuLx(i9+1) + '" y2="' + (y+bandH/2) + '" stroke="' + bandColor + '" stroke-width="1" stroke-dasharray="3,2" opacity="0.5"/>';
    }
    y += bandH;
    for (var i10 = 0; i10 < n; i10++) s += '<line x1="' + gpuCx(i10) + '" y1="' + y + '" x2="' + gpuCx(i10) + '" y2="' + (y+arrowH-4) + '" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#tc-moe)"/>';
    y += arrowH;
  
    // Row 3: next layer
    for (var i11 = 0; i11 < n; i11++) {
      s += '<rect x="' + gpuLx(i11) + '" y="' + y + '" width="' + boxW + '" height="' + boxH + '" rx="5" fill="#f1f5f9" stroke="#94a3b8" stroke-width="0.6"/>';
      s += '<text x="' + gpuCx(i11) + '" y="' + (y + boxH/2 + 1) + '" text-anchor="middle" dominant-baseline="central" font-size="12" fill="#6b7280">next layer</text>';
    }
  
    s += '</svg></div>';
    return s;
  }

  // Exports
  A.renderDeviceTopology = renderDeviceTopology;
  A.topoExpandedLayers = topoExpandedLayers;
})();
