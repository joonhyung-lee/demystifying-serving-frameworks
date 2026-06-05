/* Model Architectures tab — main module.
 * D3 SVG diagram renderer, tab initialization, model grid.
 * Depends on: model-arch-data.js, model-arch-memory.js, model-arch-topology.js
 * (loaded via ArchLib namespace).
 */
(function () {
  "use strict";
  var A = window.ArchLib;

  // Import from ArchLib
  var MODEL_ARCHITECTURES = A.MODEL_ARCHITECTURES;
  var formatParamCount = A.formatParamCount;
  var formatContextWindow = A.formatContextWindow;
  var getAttentionLabel = A.getAttentionLabel;
  var escHtml = A.escHtml;
  var getFFNSubBlocks = A.getFFNSubBlocks;
  var getAttentionSubBlocks = A.getAttentionSubBlocks;
  var computeLayerMemory = A.computeLayerMemory;
  var BLOCK_COLORS = A.BLOCK_COLORS;
  var SUB_BLOCK_COLORS = A.SUB_BLOCK_COLORS;
  var fmtBytes = A.fmtBytes;
  var GB = A.GB;
  var GPU_DEVICES = A.GPU_DEVICES;
  var WEIGHT_DTYPES = A.WEIGHT_DTYPES;
  var KV_DTYPES = A.KV_DTYPES;
  var computeMemory = A.computeMemory;
  var fmtGB = A.fmtGB;
  var defaultWeightDtypeId = A.defaultWeightDtypeId;
  var renderMemoryPanel = A.renderMemoryPanel;
  var buildVllmInternals = A.buildVllmInternals;
  var MEM_SEGMENTS = A.MEM_SEGMENTS;

  // Topology (loaded after this file, but called lazily so OK)
  function renderDeviceTopology() {
    if (A.renderDeviceTopology) A.renderDeviceTopology();
  }

  // D3 SVG Architecture Diagram Renderer
  // =========================================================================
  
  function renderDiagram(svgEl, arch, expandedBlocks, onBlockClick) {
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
  
    const containerWidth = svgEl.parentElement
      ? svgEl.parentElement.clientWidth
      : 600;
    // Diagram blocks use the left 640px; memory labels go in the right margin
    const diagramW = Math.min(containerWidth, 640);
    var memLabelOffset = 180; // extra width for memory labels on the right
    const width = diagramW + memLabelOffset;
  
    // Pre-compute layer memory sizes
    var mem = computeLayerMemory(arch);
  
    const fg = "#131416";
    const mutedFg = "#6b7280";
    const borderColor = "#d1d5db";
    const bgSubtle = "#f9fafb";
    const expandedBg = "#f1f5f9";
  
    const pad = { top: 20, right: 16, bottom: 16, left: 30 };
    const bw = diagramW - pad.left - pad.right;
    const innerW = bw - 32;
    const innerX = pad.left + 16;
    const cx = diagramW / 2;
    // X position for memory size annotations (right of diagram blocks)
    const memX = diagramW + 8;
    const blockH = 52;
    const smallH = 32;
    const arrowH = 20;
    const circleR = 9;
    const mergeGap = 36;
    const residLeftX = pad.left - 8;
    // Transformer container box — wide enough to encompass the residual bypass
    const txContainerX = residLeftX - 8;
    const txContainerW = (pad.left + bw + 4) - txContainerX;
    const collapsedTxH = 64;
    const expertSize = 36;
    const expertGap = 6;
    const expertGridH = 70;
    const subBlockH = 34;
    const subArrowH = 12;
    const subPadY = 10;
  
    const isMoE = arch.architectureType === "moe";
    const hasDenseLayers = isMoE && (arch.denseFFNLayers || 0) > 0;
    const denseLayerCount = arch.denseFFNLayers || 0;
    const moeLayerCount = (arch.numLayers || 0) - denseLayerCount;
    const isAttnExpandable =
      arch.attentionExpandable !== false &&
      arch.attentionType !== "MLA" &&
      arch.attentionType !== "AlternatingSinkGQA";
    const hasAlternatingLayers = Boolean(
      arch.alternatingLayers && arch.alternatingLayers.length > 0
    );
    const alternatingSpecs = arch.alternatingLayers || [];
  
    const attnFlow = isAttnExpandable ? getAttentionSubBlocks(arch) : null;
    const ffnFlow = getFFNSubBlocks(arch);
    const denseFFNFlow = hasDenseLayers
      ? getFFNSubBlocks(arch, { useDenseFFNDim: true })
      : null;
  
    const attnExpanded = isAttnExpandable && expandedBlocks.has("attention");
    const ffnExpanded = expandedBlocks.has(isMoE ? "experts" : "ffn");
    const txExpanded = expandedBlocks.has("transformer");
    const denseTxExpanded = expandedBlocks.has("denseTransformer");
    const denseAttnExpanded =
      isAttnExpandable && expandedBlocks.has("denseAttention");
    const denseFFNExpanded = expandedBlocks.has("denseFFN");
    const altBlockExpanded = [
      hasAlternatingLayers && expandedBlocks.has("altBlock0"),
      hasAlternatingLayers && expandedBlocks.has("altBlock1"),
    ];
    const altExpertsExpanded = [
      hasAlternatingLayers && expandedBlocks.has("altExperts0"),
      hasAlternatingLayers && expandedBlocks.has("altExperts1"),
    ];
  
    function getFlowHeight(flow, hasLabel) {
      if (flow.layout === "sequential") {
        return (
          flow.blocks.length * subBlockH +
          Math.max(0, flow.blocks.length - 1) * subArrowH +
          subPadY * 2 +
          (hasLabel ? 24 : 0)
        );
      }
      if (flow.layout === "threeWay") {
        const maxRows = Math.max(
          flow.leftPath.length,
          flow.middlePath.length,
          flow.rightPath.length
        );
        const pathLabelsH =
          flow.leftLabel || flow.middleLabel || flow.rightLabel ? 16 : 0;
        const splitH = subArrowH;
        const hasIntermediate = flow.intermediateMergeBlocks.length > 0;
        return (
          subPadY * 2 +
          (hasLabel ? 24 : 0) +
          pathLabelsH +
          splitH +
          maxRows * subBlockH +
          Math.max(0, maxRows - 1) * subArrowH +
          (subArrowH + 4) +
          (hasIntermediate
            ? flow.intermediateMergeBlocks.length * subBlockH +
              Math.max(0, flow.intermediateMergeBlocks.length - 1) *
                subArrowH +
              (subArrowH + 4)
            : 0) +
          flow.finalMergeBlocks.length * subBlockH +
          Math.max(0, flow.finalMergeBlocks.length - 1) * subArrowH
        );
      }
      // parallel
      const maxRows = Math.max(
        flow.leftPath.length,
        flow.rightPath.length
      );
      const pathLabelsH = flow.leftLabel || flow.rightLabel ? 16 : 0;
      const splitH = subArrowH;
      return (
        subPadY * 2 +
        (hasLabel ? 24 : 0) +
        pathLabelsH +
        splitH +
        maxRows * subBlockH +
        Math.max(0, maxRows - 1) * subArrowH +
        (subArrowH + 4) +
        flow.mergeBlocks.length * subBlockH +
        Math.max(0, flow.mergeBlocks.length - 1) * subArrowH
      );
    }
  
    const attnExpandedH =
      attnExpanded && attnFlow ? getFlowHeight(attnFlow, false) : 0;
    const ffnExpandedH = ffnExpanded ? getFlowHeight(ffnFlow, true) : 0;
    const denseAttnExpandedH =
      denseAttnExpanded && attnFlow ? getFlowHeight(attnFlow, false) : 0;
    const denseFFNExpandedH =
      denseFFNExpanded && denseFFNFlow
        ? getFlowHeight(denseFFNFlow, true)
        : 0;
    const altExpertsExpandedH = [
      altExpertsExpanded[0] ? getFlowHeight(ffnFlow, true) : 0,
      altExpertsExpanded[1] ? getFlowHeight(ffnFlow, true) : 0,
    ];
  
    // ---- Compute vertical positions ----
    let y = pad.top;
    const titleY = y;
    y += 44;
    const embedY = y;
    y += blockH + arrowH;
  
    // Dense transformer block
    let denseTxStart = 0,
      denseNorm1Y = 0,
      denseAttnY = 0,
      denseAttnExpandedStartY = 0,
      denseMerge1Y = 0,
      denseNorm2Y = 0,
      denseFFNBlockY = 0,
      denseFFNExpandedStartY = 0,
      denseMerge2Y = 0,
      denseTxEnd = 0;
  
    if (hasDenseLayers) {
      denseTxStart = y;
      if (denseTxExpanded) {
        y += 14;
        denseNorm1Y = y;
        y += smallH + arrowH;
        denseAttnY = y;
        y += blockH;
        denseAttnExpandedStartY = y;
        if (denseAttnExpanded) y += denseAttnExpandedH;
        y += 4;
        denseMerge1Y = y + mergeGap / 2;
        y += mergeGap;
        y += arrowH;
        denseNorm2Y = y;
        y += smallH + arrowH;
        denseFFNBlockY = y;
        y += blockH;
        denseFFNExpandedStartY = y;
        if (denseFFNExpanded) y += denseFFNExpandedH;
        denseMerge2Y = y + mergeGap / 2;
        y += mergeGap;
        y += 14;
      } else {
        y += collapsedTxH;
      }
      denseTxEnd = y;
      y += arrowH;
    }
  
    // Alternating blocks
    const altBlockStart = [0, 0],
      altBlockEnd = [0, 0];
    const altNorm1Y = [0, 0],
      altAttnY = [0, 0],
      altMerge1Y = [0, 0];
    const altNorm2Y = [0, 0],
      altRouterY = [0, 0],
      altExpertY = [0, 0];
    const altFFNExpandedStartY = [0, 0],
      altMerge2Y = [0, 0];
    let altIndicatorY = 0;
  
    const txStart = hasAlternatingLayers ? 0 : y;
    let attnY = 0,
      attnExpandedStartY = 0,
      merge1Y = 0,
      norm1Y = 0;
    let routerY = 0,
      expertY = 0,
      ffnY = 0,
      ffnExpandedStartY = 0,
      merge2Y = 0,
      norm2Y = 0;
    let txEnd = 0;
  
    if (hasAlternatingLayers) {
      for (let bi = 0; bi < 2; bi++) {
        altBlockStart[bi] = y;
        const isExp = altBlockExpanded[bi];
        const isExpExperts = altExpertsExpanded[bi];
        if (isExp) {
          y += 14;
          altNorm1Y[bi] = y;
          y += smallH + arrowH;
          altAttnY[bi] = y;
          y += blockH;
          y += 4;
          altMerge1Y[bi] = y + mergeGap / 2;
          y += mergeGap;
          y += arrowH;
          altNorm2Y[bi] = y;
          y += smallH + arrowH;
          altRouterY[bi] = y;
          y += blockH + arrowH;
          altExpertY[bi] = y;
          y += expertGridH;
          altFFNExpandedStartY[bi] = y;
          if (isExpExperts) y += altExpertsExpandedH[bi];
          altMerge2Y[bi] = y + mergeGap / 2;
          y += mergeGap;
          y += 14;
        } else {
          y += collapsedTxH;
        }
        altBlockEnd[bi] = y;
        if (bi === 0) {
          y += 10;
          altIndicatorY = y + 10;
          y += 38;
        }
      }
    } else if (txExpanded) {
      y += 14;
      norm1Y = y;
      y += smallH + arrowH;
      attnY = y;
      y += blockH;
      attnExpandedStartY = y;
      if (attnExpanded) y += attnExpandedH;
      y += 4;
      merge1Y = y + mergeGap / 2;
      y += mergeGap;
      y += arrowH;
      norm2Y = y;
      y += smallH + arrowH;
      if (isMoE) {
        routerY = y;
        y += blockH + arrowH;
        expertY = y;
        y += expertGridH;
      } else {
        ffnY = y;
        y += blockH;
      }
      ffnExpandedStartY = y;
      if (ffnExpanded) y += ffnExpandedH;
      merge2Y = y + mergeGap / 2;
      y += mergeGap;
      y += 14;
    } else {
      y += collapsedTxH;
    }
    txEnd = y;
    y += arrowH;
  
    const finalNormY = y;
    y += smallH + arrowH;
    const outputY = y;
    y += blockH + 16;
    const specsY = y;
    const specsH = 56;
    y += specsH;
    const totalH = y + pad.bottom;
  
    svg
      .attr("width", width)
      .attr("height", totalH)
      .attr("viewBox", "0 0 " + width + " " + totalH)
      .style("font-family", "'Montserrat', sans-serif");
  
    // Defs
    const defs = svg.append("defs");
    defs
      .append("marker")
      .attr("id", "arch-arrow")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 5)
      .attr("refY", 5)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .attr("orient", "auto-start-reverse")
      .append("path")
      .attr("d", "M 0 0 L 10 5 L 0 10 z")
      .attr("fill", mutedFg);
  
    defs
      .append("marker")
      .attr("id", "arch-arrow-sub")
      .attr("viewBox", "0 0 10 10")
      .attr("refX", 5)
      .attr("refY", 5)
      .attr("markerWidth", 4)
      .attr("markerHeight", 4)
      .attr("orient", "auto-start-reverse")
      .append("path")
      .attr("d", "M 0 0 L 10 5 L 0 10 z")
      .attr("fill", mutedFg);
  
    const bgG = svg.append("g");
    const g = svg.append("g");
  
    // ---- Drawing helpers ----
    /** Draw a memory size annotation to the right of a block */
    /** Draw a prominent block-total memory label above the container */
    function drawMemBlockTotal(by, text, sub) {
      g.append("text")
        .attr("x", memX)
        .attr("y", by - 6)
        .attr("text-anchor", "start")
        .attr("dominant-baseline", "central")
        .attr("fill", "#111827")
        .attr("font-size", "13px")
        .attr("font-weight", 800)
        .attr("font-family", "'Montserrat', monospace")
        .text(text);
      if (sub) {
        g.append("text")
          .attr("x", memX + 1)
          .attr("y", by + 8)
          .attr("text-anchor", "start")
          .attr("dominant-baseline", "central")
          .attr("fill", mutedFg)
          .attr("font-size", "9px")
          .attr("font-weight", 500)
          .attr("font-family", "inherit")
          .text(sub);
      }
    }
  
    /** Draw a memory size annotation to the right of a block */
    function drawMemLabel(by, bh, text, sub) {
      var ly = by + bh / 2;
      g.append("text")
        .attr("x", memX)
        .attr("y", ly - (sub ? 5 : 0))
        .attr("text-anchor", "start")
        .attr("dominant-baseline", "central")
        .attr("fill", "#374151")
        .attr("font-size", "12px")
        .attr("font-weight", 600)
        .attr("font-family", "'Montserrat', monospace")
        .text(text);
      if (sub) {
        g.append("text")
          .attr("x", memX)
          .attr("y", ly + 9)
          .attr("text-anchor", "start")
          .attr("dominant-baseline", "central")
          .attr("fill", mutedFg)
          .attr("font-size", "9px")
          .attr("font-family", "inherit")
          .text(sub);
      }
    }
  
    function drawArrow(fromY, toY) {
      g.append("line")
        .attr("x1", cx)
        .attr("y1", fromY)
        .attr("x2", cx)
        .attr("y2", toY - 2)
        .attr("stroke", mutedFg)
        .attr("stroke-width", 1.5)
        .attr("marker-end", "url(#arch-arrow)");
    }
  
    function drawResidualBypass(branchY, mY) {
      // Branch from the vertical middle of the norm block, exiting its left edge
      var midY = branchY + smallH / 2;
      var startX = innerX - 4;       // just outside the inner block's left edge
      bgG
        .append("path")
        .attr(
          "d",
          "M " +
            startX +
            " " +
            midY +
            " L " +
            residLeftX +
            " " +
            midY +
            " L " +
            residLeftX +
            " " +
            mY +
            " L " +
            (cx - circleR) +
            " " +
            mY
        )
        .attr("fill", "none")
        .attr("stroke", mutedFg)
        .attr("stroke-width", 1.5)
        .attr("opacity", 0.6);
      g.append("circle")
        .attr("cx", cx)
        .attr("cy", mY)
        .attr("r", circleR)
        .attr("fill", bgSubtle)
        .attr("stroke", mutedFg)
        .attr("stroke-width", 1.5);
      g.append("text")
        .attr("x", cx)
        .attr("y", mY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", fg)
        .attr("font-size", "14px")
        .attr("font-weight", 700)
        .attr("font-family", "inherit")
        .text("+");
    }
  
    function drawBlock(x, by, w, h, type, mainText, subText) {
      var c = BLOCK_COLORS[type];
      g.append("rect")
        .attr("x", x)
        .attr("y", by)
        .attr("width", w)
        .attr("height", h)
        .attr("rx", 8)
        .attr("fill", c.fill)
        .attr("stroke", c.stroke)
        .attr("stroke-width", 1.5);
      g.append("text")
        .attr("x", x + w / 2)
        .attr("y", by + h / 2 - (subText ? 7 : 0))
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", fg)
        .attr("font-size", "13px")
        .attr("font-weight", 600)
        .attr("font-family", "inherit")
        .text(mainText);
      if (subText) {
        g.append("text")
          .attr("x", x + w / 2)
          .attr("y", by + h / 2 + 10)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("fill", mutedFg)
          .attr("font-size", "11px")
          .attr("font-family", "inherit")
          .text(subText);
      }
    }
  
    function drawExpandableBlock(
      x,
      by,
      w,
      h,
      type,
      mainText,
      subText,
      isBlockExpanded,
      blockId
    ) {
      var c = BLOCK_COLORS[type];
      // Header region is always blockH tall; the box may be taller when expanded
      var headerMidY = by + blockH / 2;
      g.append("rect")
        .attr("x", x)
        .attr("y", by)
        .attr("width", w)
        .attr("height", h)
        .attr("rx", 8)
        .attr("fill", c.fill)
        .attr("stroke", c.stroke)
        .attr("stroke-width", isBlockExpanded ? 2.5 : 1.5);
      g.append("text")
        .attr("x", x + w / 2 - 8)
        .attr("y", headerMidY - (subText ? 7 : 0))
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", fg)
        .attr("font-size", "13px")
        .attr("font-weight", 600)
        .attr("font-family", "inherit")
        .style("pointer-events", "none")
        .text(mainText);
      if (subText) {
        g.append("text")
          .attr("x", x + w / 2 - 8)
          .attr("y", headerMidY + 10)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("fill", mutedFg)
          .attr("font-size", "11px")
          .attr("font-family", "inherit")
          .style("pointer-events", "none")
          .text(subText);
      }
      var iconX = x + w - 22;
      g.append("circle")
        .attr("cx", iconX)
        .attr("cy", headerMidY)
        .attr("r", 10)
        .attr("fill", "rgba(0,0,0,0.04)")
        .attr("stroke", c.stroke)
        .attr("stroke-width", 1)
        .style("pointer-events", "none");
      g.append("text")
        .attr("x", iconX)
        .attr("y", headerMidY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", c.stroke)
        .attr("font-size", "14px")
        .attr("font-weight", 700)
        .style("pointer-events", "none")
        .text(isBlockExpanded ? "\u2212" : "+");
      // Click target covers the header area only
      g.append("rect")
        .attr("x", x)
        .attr("y", by)
        .attr("width", w)
        .attr("height", blockH)
        .attr("rx", 8)
        .attr("fill", "transparent")
        .style("cursor", "pointer")
        .on("click", function () {
          onBlockClick(blockId);
        });
    }
  
    function drawCollapsedTransformerBlock(x, by, w, h, label, subtitle, blockId) {
      g.append("rect")
        .attr("x", x)
        .attr("y", by)
        .attr("width", w)
        .attr("height", h)
        .attr("rx", 10)
        .attr("fill", bgSubtle)
        .attr("stroke", borderColor)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "6,3");
      g.append("text")
        .attr("x", x + w / 2 - 8)
        .attr("y", by + h / 2 - 8)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", fg)
        .attr("font-size", "13px")
        .attr("font-weight", 600)
        .attr("font-family", "inherit")
        .style("pointer-events", "none")
        .text(label);
      g.append("text")
        .attr("x", x + w / 2 - 8)
        .attr("y", by + h / 2 + 10)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", mutedFg)
        .attr("font-size", "11px")
        .attr("font-family", "inherit")
        .style("pointer-events", "none")
        .text(subtitle);
      var iconX = x + w - 22,
        iconY = by + h / 2;
      g.append("circle")
        .attr("cx", iconX)
        .attr("cy", iconY)
        .attr("r", 10)
        .attr("fill", "rgba(0,0,0,0.04)")
        .attr("stroke", borderColor)
        .attr("stroke-width", 1)
        .style("pointer-events", "none");
      g.append("text")
        .attr("x", iconX)
        .attr("y", iconY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", mutedFg)
        .attr("font-size", "14px")
        .attr("font-weight", 700)
        .style("pointer-events", "none")
        .text("+");
      g.append("rect")
        .attr("x", x)
        .attr("y", by)
        .attr("width", w)
        .attr("height", h)
        .attr("rx", 10)
        .attr("fill", "transparent")
        .style("cursor", "pointer")
        .on("click", function () {
          onBlockClick(blockId);
        });
    }
  
    // Sub-block drawing
    function drawSingleSubBlock(block, bx, by, subBw, fontSize) {
      fontSize = fontSize || { name: "11px", detail: "9px" };
      var color = SUB_BLOCK_COLORS[block.type];
      g.append("rect")
        .attr("x", bx)
        .attr("y", by)
        .attr("width", subBw)
        .attr("height", subBlockH)
        .attr("rx", 6)
        .attr("fill", color.fill)
        .attr("stroke", color.stroke)
        .attr("stroke-width", 1);
      g.append("text")
        .attr("x", bx + subBw / 2)
        .attr("y", by + subBlockH / 2 - (block.detail ? 5 : 0))
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", fg)
        .attr("font-size", fontSize.name)
        .attr("font-weight", 500)
        .attr("font-family", "inherit")
        .text(block.name);
      if (block.detail) {
        g.append("text")
          .attr("x", bx + subBw / 2)
          .attr("y", by + subBlockH / 2 + 8)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("fill", mutedFg)
          .attr("font-size", fontSize.detail)
          .attr("font-family", "inherit")
          .text(block.detail);
      }
    }
  
    function drawParallelFlow(flow, startY, x, w, label, parentType, noContainer) {
      var pc = parentType && BLOCK_COLORS[parentType] ? BLOCK_COLORS[parentType] : null;
      var flowBg = pc ? pc.expandBg : expandedBg;
      var flowBorder = pc ? pc.stroke : borderColor;
      var sy = startY;
      var flowH = getFlowHeight(flow, Boolean(label));
      if (!noContainer) {
        g.append("rect")
          .attr("x", x + 4)
          .attr("y", sy)
          .attr("width", w - 8)
          .attr("height", flowH)
          .attr("rx", 8)
          .attr("fill", flowBg)
          .attr("stroke", flowBorder)
          .attr("stroke-width", 1)
          .attr("stroke-dasharray", "4,3");
        sy += subPadY;
      } else {
        // Inside parent box — no extra top margin
      }
      if (label) {
        var labelGap = noContainer ? 4 : 8;
        var labelAdvance = noContainer ? 16 : 24;
        g.append("text")
          .attr("x", x + w / 2)
          .attr("y", sy + labelGap + 4)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("fill", mutedFg)
          .attr("font-size", "10px")
          .attr("font-weight", 600)
          .attr("font-family", "inherit")
          .attr("letter-spacing", "0.5px")
          .text(label.toUpperCase());
        sy += labelAdvance;
      }
  
      var colGap = 10,
        innerPadCol = 12;
      var colW = (w - 2 * innerPadCol - colGap) / 2;
      var leftX = x + innerPadCol;
      var rightX = x + innerPadCol + colW + colGap;
      var leftCx = leftX + colW / 2;
      var rightCx = rightX + colW / 2;
      var mergeCx = x + w / 2;
  
      if (flow.leftLabel || flow.rightLabel) {
        if (flow.leftLabel) {
          g.append("text")
            .attr("x", leftCx)
            .attr("y", sy + 6)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .attr("fill", mutedFg)
            .attr("font-size", "9px")
            .attr("font-weight", 600)
            .text(flow.leftLabel);
        }
        if (flow.rightLabel) {
          g.append("text")
            .attr("x", rightCx)
            .attr("y", sy + 6)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .attr("fill", mutedFg)
            .attr("font-size", "9px")
            .attr("font-weight", 600)
            .text(flow.rightLabel);
        }
        sy += 16;
      }
      var splitTopY = sy,
        splitMidY = sy + subArrowH / 2;
      sy += subArrowH;
      var parallelStartY = sy;
      var colFontSize = { name: "10px", detail: "8px" };
  
      // split lines
      g.append("line")
        .attr("x1", mergeCx)
        .attr("y1", splitTopY)
        .attr("x2", mergeCx)
        .attr("y2", splitMidY)
        .attr("stroke", mutedFg)
        .attr("stroke-width", 1);
      g.append("path")
        .attr(
          "d",
          "M " +
            mergeCx +
            " " +
            splitMidY +
            " L " +
            leftCx +
            " " +
            splitMidY +
            " L " +
            leftCx +
            " " +
            (parallelStartY - 2)
        )
        .attr("fill", "none")
        .attr("stroke", mutedFg)
        .attr("stroke-width", 1)
        .attr("marker-end", "url(#arch-arrow-sub)");
      g.append("path")
        .attr(
          "d",
          "M " +
            mergeCx +
            " " +
            splitMidY +
            " L " +
            rightCx +
            " " +
            splitMidY +
            " L " +
            rightCx +
            " " +
            (parallelStartY - 2)
        )
        .attr("fill", "none")
        .attr("stroke", mutedFg)
        .attr("stroke-width", 1)
        .attr("marker-end", "url(#arch-arrow-sub)");
  
      // left column
      var lsy = parallelStartY;
      for (var i = 0; i < flow.leftPath.length; i++) {
        drawSingleSubBlock(flow.leftPath[i], leftX, lsy, colW, colFontSize);
        lsy += subBlockH;
        if (i < flow.leftPath.length - 1) {
          g.append("line")
            .attr("x1", leftCx)
            .attr("y1", lsy + 1)
            .attr("x2", leftCx)
            .attr("y2", lsy + subArrowH - 2)
            .attr("stroke", mutedFg)
            .attr("stroke-width", 1)
            .attr("marker-end", "url(#arch-arrow-sub)");
          lsy += subArrowH;
        }
      }
      var leftEndY = lsy;
  
      // right column
      var rsy = parallelStartY;
      for (var i = 0; i < flow.rightPath.length; i++) {
        drawSingleSubBlock(flow.rightPath[i], rightX, rsy, colW, colFontSize);
        rsy += subBlockH;
        if (i < flow.rightPath.length - 1) {
          g.append("line")
            .attr("x1", rightCx)
            .attr("y1", rsy + 1)
            .attr("x2", rightCx)
            .attr("y2", rsy + subArrowH - 2)
            .attr("stroke", mutedFg)
            .attr("stroke-width", 1)
            .attr("marker-end", "url(#arch-arrow-sub)");
          rsy += subArrowH;
        }
      }
      var rightEndY = rsy;
  
      var maxRows = Math.max(flow.leftPath.length, flow.rightPath.length);
      var mergeStartY =
        parallelStartY +
        maxRows * subBlockH +
        Math.max(0, maxRows - 1) * subArrowH +
        subArrowH +
        4;
      var subInnerXLocal = x + 16;
      var subInnerWLocal = w - 40;
      var firstIsCircle = flow.mergeBlocks[0] && flow.mergeBlocks[0].circleSymbol;
  
      if (firstIsCircle) {
        var circleCy = mergeStartY + subBlockH / 2;
        g.append("path")
          .attr(
            "d",
            "M " +
              leftCx +
              " " +
              (leftEndY + 1) +
              " L " +
              leftCx +
              " " +
              circleCy +
              " L " +
              (mergeCx - circleR - 2) +
              " " +
              circleCy
          )
          .attr("fill", "none")
          .attr("stroke", mutedFg)
          .attr("stroke-width", 1);
        g.append("path")
          .attr(
            "d",
            "M " +
              rightCx +
              " " +
              (rightEndY + 1) +
              " L " +
              rightCx +
              " " +
              circleCy +
              " L " +
              (mergeCx + circleR + 2) +
              " " +
              circleCy
          )
          .attr("fill", "none")
          .attr("stroke", mutedFg)
          .attr("stroke-width", 1);
      } else {
        var convergeHorizY =
          Math.max(leftEndY, rightEndY) + (subArrowH + 4) / 2;
        g.append("path")
          .attr(
            "d",
            "M " +
              leftCx +
              " " +
              (leftEndY + 1) +
              " L " +
              leftCx +
              " " +
              convergeHorizY +
              " L " +
              mergeCx +
              " " +
              convergeHorizY
          )
          .attr("fill", "none")
          .attr("stroke", mutedFg)
          .attr("stroke-width", 1);
        g.append("path")
          .attr(
            "d",
            "M " +
              rightCx +
              " " +
              (rightEndY + 1) +
              " L " +
              rightCx +
              " " +
              convergeHorizY +
              " L " +
              mergeCx +
              " " +
              convergeHorizY
          )
          .attr("fill", "none")
          .attr("stroke", mutedFg)
          .attr("stroke-width", 1);
        g.append("line")
          .attr("x1", mergeCx)
          .attr("y1", convergeHorizY)
          .attr("x2", mergeCx)
          .attr("y2", mergeStartY - 2)
          .attr("stroke", mutedFg)
          .attr("stroke-width", 1)
          .attr("marker-end", "url(#arch-arrow-sub)");
      }
  
      var msy = mergeStartY;
      for (var i = 0; i < flow.mergeBlocks.length; i++) {
        var block = flow.mergeBlocks[i];
        if (block.circleSymbol) {
          var circleCy2 = msy + subBlockH / 2;
          var symbolR = circleR + 2;
          g.append("circle")
            .attr("cx", mergeCx)
            .attr("cy", circleCy2)
            .attr("r", symbolR)
            .attr("fill", bgSubtle)
            .attr("stroke", mutedFg)
            .attr("stroke-width", 1.5);
          g.append("text")
            .attr("x", mergeCx)
            .attr("y", circleCy2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .attr("fill", fg)
            .attr("font-size", "14px")
            .attr("font-weight", 700)
            .attr("font-family", "inherit")
            .text(block.circleSymbol);
        } else {
          drawSingleSubBlock(block, subInnerXLocal, msy, subInnerWLocal);
        }
        msy += subBlockH;
        if (i < flow.mergeBlocks.length - 1) {
          g.append("line")
            .attr("x1", mergeCx)
            .attr("y1", msy + 1)
            .attr("x2", mergeCx)
            .attr("y2", msy + subArrowH - 2)
            .attr("stroke", mutedFg)
            .attr("stroke-width", 1)
            .attr("marker-end", "url(#arch-arrow-sub)");
          msy += subArrowH;
        }
      }
    }
  
    function drawThreeWayFlow(flow, startY, x, w, parentType, noContainer) {
      var pc = parentType && BLOCK_COLORS[parentType] ? BLOCK_COLORS[parentType] : null;
      var flowBg = pc ? pc.expandBg : expandedBg;
      var flowBorder = pc ? pc.stroke : borderColor;
      var sy = startY;
      var flowH = getFlowHeight(flow, false);
      if (!noContainer) {
        g.append("rect")
          .attr("x", x + 4)
          .attr("y", sy)
          .attr("width", w - 8)
          .attr("height", flowH)
          .attr("rx", 8)
          .attr("fill", flowBg)
          .attr("stroke", flowBorder)
          .attr("stroke-width", 1)
          .attr("stroke-dasharray", "4,3");
        sy += subPadY;
      }
      // noContainer: no extra top padding — content starts immediately
  
      var colGap = 6,
        innerPadCol = 8;
      var colW = (w - 2 * innerPadCol - 2 * colGap) / 3;
      var leftX = x + innerPadCol;
      var middleX = x + innerPadCol + colW + colGap;
      var rightX = x + innerPadCol + 2 * (colW + colGap);
      var leftCx = leftX + colW / 2;
      var middleCx = middleX + colW / 2;
      var rightCx = rightX + colW / 2;
      var mergeCxLocal = x + w / 2;
  
      if (flow.leftLabel || flow.middleLabel || flow.rightLabel) {
        [
          { label: flow.leftLabel, cx: leftCx },
          { label: flow.middleLabel, cx: middleCx },
          { label: flow.rightLabel, cx: rightCx },
        ].forEach(function (item) {
          if (item.label) {
            g.append("text")
              .attr("x", item.cx)
              .attr("y", sy + 6)
              .attr("text-anchor", "middle")
              .attr("dominant-baseline", "central")
              .attr("fill", mutedFg)
              .attr("font-size", "9px")
              .attr("font-weight", 600)
              .text(item.label);
          }
        });
        sy += 16;
      }
  
      var splitTopY = sy,
        splitMidY = sy + subArrowH / 2;
      sy += subArrowH;
      var parallelStartY = sy;
      var colFontSize = { name: "9px", detail: "7.5px" };
  
      g.append("line")
        .attr("x1", mergeCxLocal)
        .attr("y1", splitTopY)
        .attr("x2", mergeCxLocal)
        .attr("y2", splitMidY)
        .attr("stroke", mutedFg)
        .attr("stroke-width", 1);
      [leftCx, middleCx, rightCx].forEach(function (colCx) {
        g.append("path")
          .attr(
            "d",
            "M " +
              mergeCxLocal +
              " " +
              splitMidY +
              " L " +
              colCx +
              " " +
              splitMidY +
              " L " +
              colCx +
              " " +
              (parallelStartY - 2)
          )
          .attr("fill", "none")
          .attr("stroke", mutedFg)
          .attr("stroke-width", 1)
          .attr("marker-end", "url(#arch-arrow-sub)");
      });
  
      // Draw columns
      var colEndY = [0, 0, 0];
      [
        { path: flow.leftPath, bx: leftX, colCx: leftCx },
        { path: flow.middlePath, bx: middleX, colCx: middleCx },
        { path: flow.rightPath, bx: rightX, colCx: rightCx },
      ].forEach(function (col, ci) {
        var csy = parallelStartY;
        for (var i = 0; i < col.path.length; i++) {
          drawSingleSubBlock(col.path[i], col.bx, csy, colW, colFontSize);
          csy += subBlockH;
          if (i < col.path.length - 1) {
            g.append("line")
              .attr("x1", col.colCx)
              .attr("y1", csy + 1)
              .attr("x2", col.colCx)
              .attr("y2", csy + subArrowH - 2)
              .attr("stroke", mutedFg)
              .attr("stroke-width", 1)
              .attr("marker-end", "url(#arch-arrow-sub)");
            csy += subArrowH;
          }
        }
        colEndY[ci] = csy;
      });
  
      var maxRows = Math.max(
        flow.leftPath.length,
        flow.middlePath.length,
        flow.rightPath.length
      );
      var afterPathsY =
        parallelStartY +
        maxRows * subBlockH +
        Math.max(0, maxRows - 1) * subArrowH;
      var convergeY = afterPathsY + (subArrowH + 4) / 2;
  
      // converge all three to center
      [
        { endY: colEndY[0], cx: leftCx },
        { endY: colEndY[1], cx: middleCx },
        { endY: colEndY[2], cx: rightCx },
      ].forEach(function (c) {
        g.append("path")
          .attr(
            "d",
            "M " +
              c.cx +
              " " +
              (c.endY + 1) +
              " L " +
              c.cx +
              " " +
              convergeY +
              " L " +
              mergeCxLocal +
              " " +
              convergeY
          )
          .attr("fill", "none")
          .attr("stroke", mutedFg)
          .attr("stroke-width", 1);
      });
  
      var mergeStartY = afterPathsY + subArrowH + 4;
      g.append("line")
        .attr("x1", mergeCxLocal)
        .attr("y1", convergeY)
        .attr("x2", mergeCxLocal)
        .attr("y2", mergeStartY - 2)
        .attr("stroke", mutedFg)
        .attr("stroke-width", 1)
        .attr("marker-end", "url(#arch-arrow-sub)");
  
      var subInnerXLocal = x + 16;
      var subInnerWLocal = w - 40;
      var msy = mergeStartY;
      flow.finalMergeBlocks.forEach(function (block, i) {
        drawSingleSubBlock(block, subInnerXLocal, msy, subInnerWLocal);
        msy += subBlockH;
        if (i < flow.finalMergeBlocks.length - 1) {
          g.append("line")
            .attr("x1", mergeCxLocal)
            .attr("y1", msy + 1)
            .attr("x2", mergeCxLocal)
            .attr("y2", msy + subArrowH - 2)
            .attr("stroke", mutedFg)
            .attr("stroke-width", 1)
            .attr("marker-end", "url(#arch-arrow-sub)");
          msy += subArrowH;
        }
      });
    }
  
    function drawFlow(flow, startY, x, w, label, parentType, noContainer) {
      if (flow.layout === "threeWay") drawThreeWayFlow(flow, startY, x, w, parentType, noContainer);
      else if (flow.layout === "parallel")
        drawParallelFlow(flow, startY, x, w, label, parentType, noContainer);
    }
  
    // Expert grid
    function drawExpertGrid(
      eY,
      isExpExpanded,
      expandedH,
      expandedStartY2,
      n2Y,
      m2Y,
      expertBlockId,
      layerCount
    ) {
      layerCount = layerCount || 1;
      var routedCount = arch.hasSharedExpert
        ? (arch.numExperts || 0) - 1
        : arch.numExperts;
      var routerSub =
        "Top-" +
        arch.activeExperts +
        " of " +
        routedCount +
        " routed" +
        (arch.hasSharedExpert ? " + 1 shared" : "");
      var rY = n2Y + smallH + arrowH;
      drawBlock(innerX, rY, innerW, blockH, "router", "MoE Router", routerSub);
      drawMemLabel(rY, blockH, fmtBytes(mem.routerPerLayer * layerCount),
        "= " + fmtBytes(mem.routerPerLayer) + "/layer \u00D7 " + layerCount);
      drawArrow(rY + blockH, rY + blockH + arrowH);
  
      var ec = BLOCK_COLORS.expert;
      var eac = BLOCK_COLORS.expertActive;
  
      // Single bounding box that grows when expanded
      var totalExpertH = isExpExpanded
        ? expertGridH + expandedH
        : expertGridH;
  
      g.append("rect")
        .attr("x", innerX)
        .attr("y", eY)
        .attr("width", innerW)
        .attr("height", totalExpertH)
        .attr("rx", 8)
        .attr("fill", ec.fill)
        .attr("stroke", ec.stroke)
        .attr("stroke-width", isExpExpanded ? 2.5 : 1.5)
        .attr("stroke-dasharray", "4,3");
  
      // Expert mini-boxes inside the top portion
      var numActive = arch.activeExperts || 2;
      var numShow = Math.min(arch.numExperts || 8, 6);
      var showEllipsis = (arch.numExperts || 0) > numShow;
      var totalBoxes = numShow + (showEllipsis ? 2 : 0);
      var totalBoxWidth =
        totalBoxes * expertSize + (totalBoxes - 1) * expertGap;
      var ex = cx - totalBoxWidth / 2;
      var ey = eY + (expertGridH - expertSize) / 2;
  
      for (var i = 0; i < numShow; i++) {
        var isActive = i < numActive;
        g.append("rect")
          .attr("x", ex)
          .attr("y", ey)
          .attr("width", expertSize)
          .attr("height", expertSize)
          .attr("rx", 6)
          .attr("fill", isActive ? eac.fill : bgSubtle)
          .attr("stroke", isActive ? eac.stroke : borderColor)
          .attr("stroke-width", isActive ? 1.5 : 1)
          .attr("opacity", isActive ? 1 : 0.6)
          .style("pointer-events", "none");
        g.append("text")
          .attr("x", ex + expertSize / 2)
          .attr("y", ey + expertSize / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("fill", isActive ? fg : mutedFg)
          .attr("font-size", "9px")
          .attr("font-weight", isActive ? 600 : 400)
          .attr("font-family", "inherit")
          .style("pointer-events", "none")
          .text("E" + (i + 1));
        ex += expertSize + expertGap;
      }
      if (showEllipsis) {
        g.append("text")
          .attr("x", ex + expertSize / 2)
          .attr("y", ey + expertSize / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("fill", mutedFg)
          .attr("font-size", "14px")
          .attr("font-weight", 700)
          .style("pointer-events", "none")
          .text("\u00B7\u00B7\u00B7");
        ex += expertSize + expertGap;
        g.append("rect")
          .attr("x", ex)
          .attr("y", ey + 3)
          .attr("width", expertSize)
          .attr("height", expertSize - 6)
          .attr("rx", 4)
          .attr("fill", bgSubtle)
          .attr("stroke", borderColor)
          .attr("stroke-width", 1)
          .style("pointer-events", "none");
        g.append("text")
          .attr("x", ex + expertSize / 2)
          .attr("y", ey + expertSize / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("fill", mutedFg)
          .attr("font-size", "9px")
          .attr("font-weight", 600)
          .attr("font-family", "inherit")
          .style("pointer-events", "none")
          .text("\u00D7" + arch.numExperts);
      }
  
      // expand/collapse icon
      var expIconX = innerX + innerW - 18,
        expIconY = eY + expertGridH / 2;
      g.append("circle")
        .attr("cx", expIconX)
        .attr("cy", expIconY)
        .attr("r", 10)
        .attr("fill", "rgba(0,0,0,0.04)")
        .attr("stroke", ec.stroke)
        .attr("stroke-width", 1)
        .style("pointer-events", "none");
      g.append("text")
        .attr("x", expIconX)
        .attr("y", expIconY)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", ec.stroke)
        .attr("font-size", "14px")
        .attr("font-weight", 700)
        .style("pointer-events", "none")
        .text(isExpExpanded ? "\u2212" : "+");
  
      // Click target covers the top expert-grid area
      g.append("rect")
        .attr("x", innerX)
        .attr("y", eY)
        .attr("width", innerW)
        .attr("height", expertGridH)
        .attr("rx", 8)
        .attr("fill", "transparent")
        .style("cursor", "pointer")
        .on("click", function () {
          onBlockClick(expertBlockId);
        });
  
      // Expert grid memory label
      var expertTotalMem = mem.allExpertsPerLayer + mem.sharedExpertPerLayer;
      drawMemLabel(eY, expertGridH, fmtBytes(expertTotalMem * layerCount),
        "= " + fmtBytes(expertTotalMem) + "/layer \u00D7 " + layerCount + " (all experts)");
  
      // Expanded sub-blocks drawn inside the same box (no separate container)
      if (isExpExpanded) {
        drawFlow(ffnFlow, expandedStartY2, innerX, innerW, "Expert FFN (SwiGLU)", "expert", true);
      }
  
      var expertBottom = isExpExpanded
        ? expandedStartY2 + expandedH
        : eY + expertGridH;
      drawArrow(expertBottom + 4, m2Y - circleR);
      drawResidualBypass(n2Y, m2Y);
    }
  
    // ======== RENDER BLOCKS ========
  
    // Title
    var modelName = arch.developer || "";
    var paramsSummary = [
      formatParamCount(arch.totalParams) + " total",
      isMoE ? formatParamCount(arch.activeParams) + " active" : null,
      arch.contextWindow
        ? formatContextWindow(arch.contextWindow) + " context"
        : null,
    ]
      .filter(Boolean)
      .join(" \u00B7 ");
    g.append("text")
      .attr("x", cx)
      .attr("y", titleY + 14)
      .attr("text-anchor", "middle")
      .attr("fill", fg)
      .attr("font-size", "15px")
      .attr("font-weight", 700)
      .attr("font-family", "inherit")
      .text(modelName);
    g.append("text")
      .attr("x", cx)
      .attr("y", titleY + 32)
      .attr("text-anchor", "middle")
      .attr("fill", mutedFg)
      .attr("font-size", "11px")
      .attr("font-family", "inherit")
      .text(paramsSummary);
  
    // Memory column header (right side)
    var dtypeLabel = (arch.ckptDtype || "bf16").toUpperCase();
    g.append("text")
      .attr("x", memX)
      .attr("y", titleY + 14)
      .attr("text-anchor", "start")
      .attr("fill", fg)
      .attr("font-size", "13px")
      .attr("font-weight", 700)
      .attr("font-family", "inherit")
      .text("Weight Memory");
    g.append("text")
      .attr("x", memX)
      .attr("y", titleY + 30)
      .attr("text-anchor", "start")
      .attr("fill", mutedFg)
      .attr("font-size", "10px")
      .attr("font-family", "inherit")
      .text("ckpt dtype: " + dtypeLabel + " (" + (arch.ckptBytes || 2) + "B/param)");
  
    // Embedding
    var embedSub = [
      arch.hiddenSize ? "d = " + arch.hiddenSize.toLocaleString() : null,
      arch.vocabSize ? "vocab = " + arch.vocabSize.toLocaleString() : null,
    ]
      .filter(Boolean)
      .join("  \u00B7  ");
    drawBlock(
      pad.left,
      embedY,
      bw,
      blockH,
      "embedding",
      "Token Embedding",
      embedSub || undefined
    );
    drawMemLabel(embedY, blockH, fmtBytes(mem.embedding));
    drawArrow(
      embedY + blockH,
      hasDenseLayers
        ? denseTxStart
        : hasAlternatingLayers
        ? altBlockStart[0]
        : txStart
    );
  
    // Dense transformer block
    if (hasDenseLayers) {
      if (denseTxExpanded) {
        // Dense transformer block total
        drawMemBlockTotal(denseTxStart,
          fmtBytes(mem.denseTransformerPerLayer * denseLayerCount),
          fmtBytes(mem.denseTransformerPerLayer) + "/layer \u00D7 " + denseLayerCount);
        g.append("rect")
          .attr("x", txContainerX)
          .attr("y", denseTxStart)
          .attr("width", txContainerW)
          .attr("height", denseTxEnd - denseTxStart)
          .attr("rx", 10)
          .attr("fill", "none")
          .attr("stroke", borderColor)
          .attr("stroke-width", 2)
          .attr("stroke-dasharray", "6,3");
        var denseLabel = "\u2212 \u00D7" + denseLayerCount + " dense layers";
        var denseBadgeW = denseLabel.length * 7 + 16;
        g.append("rect")
          .attr("x", diagramW - pad.right - denseBadgeW - 4)
          .attr("y", denseTxStart - 11)
          .attr("width", denseBadgeW)
          .attr("height", 22)
          .attr("rx", 11)
          .attr("fill", bgSubtle)
          .attr("stroke", borderColor)
          .attr("stroke-width", 1);
        g.append("text")
          .attr("x", diagramW - pad.right - denseBadgeW / 2 - 4)
          .attr("y", denseTxStart)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "central")
          .attr("fill", mutedFg)
          .attr("font-size", "11px")
          .attr("font-weight", 600)
          .attr("font-family", "inherit")
          .text(denseLabel);
        g.append("rect")
          .attr("x", diagramW - pad.right - denseBadgeW - 4)
          .attr("y", denseTxStart - 11)
          .attr("width", denseBadgeW)
          .attr("height", 22)
          .attr("rx", 11)
          .attr("fill", "transparent")
          .style("cursor", "pointer")
          .on("click", function () {
            onBlockClick("denseTransformer");
          });
  
        var denseN = denseLayerCount;
        drawBlock(innerX, denseNorm1Y, innerW, smallH, "norm", "RMSNorm");
        drawMemLabel(denseNorm1Y, smallH, fmtBytes(mem.normPerLayer / 2 * denseN),
          "= " + fmtBytes(mem.normPerLayer / 2) + " \u00D7 " + denseN);
        drawArrow(denseNorm1Y + smallH, denseAttnY);
        var denseAttnLabel = getAttentionLabel(arch.attentionType);
        var denseHeadSub = arch.numHeads ? arch.numHeads + " heads" : undefined;
        if (isAttnExpandable) {
          var denseAttnTotalH = denseAttnExpanded ? blockH + denseAttnExpandedH : blockH;
          drawExpandableBlock(
            innerX,
            denseAttnY,
            innerW,
            denseAttnTotalH,
            "attention",
            denseAttnLabel,
            denseHeadSub,
            denseAttnExpanded,
            "denseAttention"
          );
          if (denseAttnExpanded && attnFlow)
            drawFlow(attnFlow, denseAttnExpandedStartY, innerX, innerW, null, "attention", true);
        } else {
          drawBlock(
            innerX,
            denseAttnY,
            innerW,
            blockH,
            "attention",
            denseAttnLabel,
            denseHeadSub
          );
        }
        var denseAttnBottom = denseAttnExpanded
          ? denseAttnExpandedStartY + denseAttnExpandedH + 4
          : denseAttnY + blockH + 4;
        drawMemLabel(denseAttnY, blockH, fmtBytes(mem.attentionPerLayer * denseN),
          "= " + fmtBytes(mem.attentionPerLayer) + "/layer \u00D7 " + denseN);
        drawArrow(denseAttnBottom, denseMerge1Y - circleR);
        drawResidualBypass(denseNorm1Y, denseMerge1Y);
        drawArrow(denseMerge1Y + circleR, denseNorm2Y);
        drawBlock(innerX, denseNorm2Y, innerW, smallH, "norm", "RMSNorm");
        drawMemLabel(denseNorm2Y, smallH, fmtBytes(mem.normPerLayer / 2 * denseN),
          "= " + fmtBytes(mem.normPerLayer / 2) + " \u00D7 " + denseN);
        drawArrow(denseNorm2Y + smallH, denseFFNBlockY);
        var denseFFNSub = arch.denseFFNDim
          ? "intermediate = " + arch.denseFFNDim.toLocaleString()
          : "Dense Feed-Forward";
        var denseFFNTotalH = denseFFNExpanded
          ? blockH + denseFFNExpandedH
          : blockH;
        drawExpandableBlock(
          innerX,
          denseFFNBlockY,
          innerW,
          denseFFNTotalH,
          "ffn",
          "Dense FFN",
          denseFFNSub,
          denseFFNExpanded,
          "denseFFN"
        );
        drawMemLabel(denseFFNBlockY, blockH, fmtBytes(mem.denseFfnPerLayer * denseN),
          "= " + fmtBytes(mem.denseFfnPerLayer) + "/layer \u00D7 " + denseN);
        if (denseFFNExpanded && denseFFNFlow)
          drawFlow(
            denseFFNFlow,
            denseFFNExpandedStartY,
            innerX,
            innerW,
            "SwiGLU FFN",
            "ffn",
            true
          );
        var denseFFNBottom = denseFFNExpanded
          ? denseFFNExpandedStartY + denseFFNExpandedH + 4
          : denseFFNBlockY + blockH + 4;
        drawArrow(denseFFNBottom, denseMerge2Y - circleR);
        drawResidualBypass(denseNorm2Y, denseMerge2Y);
      } else {
        var denseSub2 =
          "\u00D7" +
          denseLayerCount +
          " dense layers" +
          (arch.denseFFNDim
            ? " \u00B7 FFN = " + arch.denseFFNDim.toLocaleString()
            : "");
        drawCollapsedTransformerBlock(
          pad.left,
          denseTxStart,
          bw,
          collapsedTxH,
          "Dense Transformer Block",
          denseSub2,
          "denseTransformer"
        );
        drawMemLabel(denseTxStart, collapsedTxH,
          fmtBytes(mem.denseTransformerPerLayer * denseLayerCount),
          fmtBytes(mem.denseTransformerPerLayer) + "/layer \u00D7 " + denseLayerCount);
      }
      drawArrow(
        denseTxEnd,
        hasAlternatingLayers ? altBlockStart[0] : txStart
      );
    }
  
    var attnLabel = getAttentionLabel(arch.attentionType);
    var mainLayerCount = hasDenseLayers ? moeLayerCount : arch.numLayers;
    var layerSuffix = hasDenseLayers ? " MoE layers" : " layers";
    var layerLabel = mainLayerCount
      ? "\u00D7" + mainLayerCount + layerSuffix
      : "Transformer Block";
  
    // Alternating blocks
    if (hasAlternatingLayers) {
      for (var bi = 0; bi < 2; bi++) {
        var spec = alternatingSpecs[bi];
        if (altBlockExpanded[bi]) {
          // Alternating block total
          drawMemBlockTotal(altBlockStart[bi],
            fmtBytes(mem.transformerPerLayer * spec.count),
            fmtBytes(mem.transformerPerLayer) + "/layer \u00D7 " + spec.count);
          g.append("rect")
            .attr("x", txContainerX)
            .attr("y", altBlockStart[bi])
            .attr("width", txContainerW)
            .attr("height", altBlockEnd[bi] - altBlockStart[bi])
            .attr("rx", 10)
            .attr("fill", "none")
            .attr("stroke", borderColor)
            .attr("stroke-width", 2)
            .attr("stroke-dasharray", "6,3");
          var altLabel2 =
            "\u2212 \u00D7" + spec.count + " " + spec.label + " layers";
          var altBadgeW = altLabel2.length * 6.5 + 16;
          g.append("rect")
            .attr("x", diagramW - pad.right - altBadgeW - 4)
            .attr("y", altBlockStart[bi] - 11)
            .attr("width", altBadgeW)
            .attr("height", 22)
            .attr("rx", 11)
            .attr("fill", bgSubtle)
            .attr("stroke", borderColor)
            .attr("stroke-width", 1);
          g.append("text")
            .attr("x", diagramW - pad.right - altBadgeW / 2 - 4)
            .attr("y", altBlockStart[bi])
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .attr("fill", mutedFg)
            .attr("font-size", "10px")
            .attr("font-weight", 600)
            .attr("font-family", "inherit")
            .text(altLabel2);
          (function (bi2) {
            g.append("rect")
              .attr("x", diagramW - pad.right - altBadgeW - 4)
              .attr("y", altBlockStart[bi2] - 11)
              .attr("width", altBadgeW)
              .attr("height", 22)
              .attr("rx", 11)
              .attr("fill", "transparent")
              .style("cursor", "pointer")
              .on("click", function () {
                onBlockClick("altBlock" + bi2);
              });
          })(bi);
  
          drawBlock(
            innerX,
            altNorm1Y[bi],
            innerW,
            smallH,
            "norm",
            "RMSNorm"
          );
          var altN = spec.count;
          drawMemLabel(altNorm1Y[bi], smallH, fmtBytes(mem.normPerLayer / 2 * altN),
            "= " + fmtBytes(mem.normPerLayer / 2) + " \u00D7 " + altN);
          drawArrow(altNorm1Y[bi] + smallH, altAttnY[bi]);
          drawBlock(
            innerX,
            altAttnY[bi],
            innerW,
            blockH,
            "attention",
            spec.label,
            spec.description ? spec.description.substring(0, 60) : undefined
          );
          drawMemLabel(altAttnY[bi], blockH, fmtBytes(mem.attentionPerLayer * altN),
            "= " + fmtBytes(mem.attentionPerLayer) + "/layer \u00D7 " + altN);
          drawArrow(altAttnY[bi] + blockH + 4, altMerge1Y[bi] - circleR);
          drawResidualBypass(altNorm1Y[bi], altMerge1Y[bi]);
          drawArrow(altMerge1Y[bi] + circleR, altNorm2Y[bi]);
          drawBlock(
            innerX,
            altNorm2Y[bi],
            innerW,
            smallH,
            "norm",
            "RMSNorm"
          );
          drawMemLabel(altNorm2Y[bi], smallH, fmtBytes(mem.normPerLayer / 2 * altN),
            "= " + fmtBytes(mem.normPerLayer / 2) + " \u00D7 " + altN);
          drawExpertGrid(
            altExpertY[bi],
            altExpertsExpanded[bi],
            altExpertsExpandedH[bi],
            altFFNExpandedStartY[bi],
            altNorm2Y[bi],
            altMerge2Y[bi],
            "altExperts" + bi,
            spec.count
          );
        } else {
          var altCollapsedSub =
            "\u00D7" +
            spec.count +
            " " +
            spec.label +
            " layers";
          drawCollapsedTransformerBlock(
            pad.left,
            altBlockStart[bi],
            bw,
            collapsedTxH,
            spec.label,
            altCollapsedSub,
            "altBlock" + bi
          );
          drawMemLabel(altBlockStart[bi], collapsedTxH,
            fmtBytes(mem.transformerPerLayer * spec.count),
            fmtBytes(mem.transformerPerLayer) + "/layer \u00D7 " + spec.count);
        }
        if (bi === 0) {
          g.append("text")
            .attr("x", cx)
            .attr("y", altIndicatorY)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "central")
            .attr("fill", mutedFg)
            .attr("font-size", "10px")
            .attr("font-style", "italic")
            .attr("font-family", "inherit")
            .text("\u21C5 alternating every layer");
        }
      }
    } else if (txExpanded) {
      // Expanded main transformer — block total label
      var mainLayerCount2 = hasDenseLayers ? moeLayerCount : (arch.numLayers || 0);
      drawMemBlockTotal(txStart,
        fmtBytes(mem.transformerPerLayer * mainLayerCount2),
        fmtBytes(mem.transformerPerLayer) + "/layer \u00D7 " + mainLayerCount2);
      g.append("rect")
        .attr("x", txContainerX)
        .attr("y", txStart)
        .attr("width", txContainerW)
        .attr("height", txEnd - txStart)
        .attr("rx", 10)
        .attr("fill", "none")
        .attr("stroke", borderColor)
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", "6,3");
      var txBadgeLabel = "\u2212 " + layerLabel;
      var txBadgeW = txBadgeLabel.length * 7 + 16;
      g.append("rect")
        .attr("x", diagramW - pad.right - txBadgeW - 4)
        .attr("y", txStart - 11)
        .attr("width", txBadgeW)
        .attr("height", 22)
        .attr("rx", 11)
        .attr("fill", bgSubtle)
        .attr("stroke", borderColor)
        .attr("stroke-width", 1);
      g.append("text")
        .attr("x", diagramW - pad.right - txBadgeW / 2 - 4)
        .attr("y", txStart)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", mutedFg)
        .attr("font-size", "11px")
        .attr("font-weight", 600)
        .attr("font-family", "inherit")
        .text(txBadgeLabel);
      g.append("rect")
        .attr("x", diagramW - pad.right - txBadgeW - 4)
        .attr("y", txStart - 11)
        .attr("width", txBadgeW)
        .attr("height", 22)
        .attr("rx", 11)
        .attr("fill", "transparent")
        .style("cursor", "pointer")
        .on("click", function () {
          onBlockClick("transformer");
        });
  
      drawBlock(innerX, norm1Y, innerW, smallH, "norm", "RMSNorm");
      var txLayerN = hasDenseLayers ? moeLayerCount : (arch.numLayers || 0);
      drawMemLabel(norm1Y, smallH, fmtBytes(mem.normPerLayer / 2 * txLayerN),
        "= " + fmtBytes(mem.normPerLayer / 2) + " \u00D7 " + txLayerN);
      drawArrow(norm1Y + smallH, attnY);
  
      var headSub = arch.numHeads ? arch.numHeads + " heads" : undefined;
      if (isAttnExpandable) {
        var attnTotalH = attnExpanded ? blockH + attnExpandedH : blockH;
        drawExpandableBlock(
          innerX,
          attnY,
          innerW,
          attnTotalH,
          "attention",
          attnLabel,
          headSub,
          attnExpanded,
          "attention"
        );
        if (attnExpanded && attnFlow)
          drawFlow(attnFlow, attnExpandedStartY, innerX, innerW, null, "attention", true);
      } else {
        drawBlock(
          innerX,
          attnY,
          innerW,
          blockH,
          "attention",
          attnLabel,
          headSub
        );
      }
      drawMemLabel(attnY, blockH, fmtBytes(mem.attentionPerLayer * txLayerN),
        "= " + fmtBytes(mem.attentionPerLayer) + "/layer \u00D7 " + txLayerN);
      var attnBottom = attnExpanded
        ? attnExpandedStartY + attnExpandedH + 4
        : attnY + blockH + 4;
      drawArrow(attnBottom, merge1Y - circleR);
      drawResidualBypass(norm1Y, merge1Y);
      drawArrow(merge1Y + circleR, norm2Y);
      drawBlock(innerX, norm2Y, innerW, smallH, "norm", "RMSNorm");
      drawMemLabel(norm2Y, smallH, fmtBytes(mem.normPerLayer / 2 * txLayerN),
        "= " + fmtBytes(mem.normPerLayer / 2) + " \u00D7 " + txLayerN);
      drawArrow(norm2Y + smallH, isMoE ? routerY : ffnY);
  
      if (isMoE) {
        var mainMoeLayers = hasDenseLayers ? moeLayerCount : (arch.numLayers || 0);
        drawExpertGrid(
          expertY,
          ffnExpanded,
          ffnExpandedH,
          ffnExpandedStartY,
          norm2Y,
          merge2Y,
          "experts",
          mainMoeLayers
        );
      } else {
        var ffnSub2 = arch.ffnDim
          ? "intermediate = " + arch.ffnDim.toLocaleString()
          : "Feed-Forward";
        var ffnTotalH = ffnExpanded ? blockH + ffnExpandedH : blockH;
        drawExpandableBlock(
          innerX,
          ffnY,
          innerW,
          ffnTotalH,
          "ffn",
          "FFN (SwiGLU)",
          ffnSub2,
          ffnExpanded,
          "ffn"
        );
        drawMemLabel(ffnY, blockH, fmtBytes(mem.denseFfnPerLayer * txLayerN),
          "= " + fmtBytes(mem.denseFfnPerLayer) + "/layer \u00D7 " + txLayerN);
        if (ffnExpanded) drawFlow(ffnFlow, ffnExpandedStartY, innerX, innerW, "SwiGLU FFN", "ffn", true);
        var ffnBottom = ffnExpanded
          ? ffnExpandedStartY + ffnExpandedH + 4
          : ffnY + blockH + 4;
        drawArrow(ffnBottom, merge2Y - circleR);
        drawResidualBypass(norm2Y, merge2Y);
      }
    } else {
      // Collapsed main transformer
      var txSub = layerLabel;
      if (isMoE)
        txSub +=
          " \u00B7 " +
          (arch.numExperts || "?") +
          " experts";
      drawCollapsedTransformerBlock(
        pad.left,
        txStart,
        bw,
        collapsedTxH,
        "Transformer Block",
        txSub,
        "transformer"
      );
      var mainLayers = hasDenseLayers ? moeLayerCount : (arch.numLayers || 0);
      drawMemLabel(txStart, collapsedTxH,
        fmtBytes(mem.transformerPerLayer * mainLayers),
        fmtBytes(mem.transformerPerLayer) + "/layer \u00D7 " + mainLayers);
    }
  
    // Final RMSNorm
    drawArrow(txEnd, finalNormY);
    drawBlock(pad.left, finalNormY, bw, smallH, "norm", "RMSNorm");
    drawArrow(finalNormY + smallH, outputY);
  
    // Output
    drawBlock(pad.left, outputY, bw, blockH, "output", "Output Head", "LM head");
    drawMemLabel(outputY, blockH, fmtBytes(mem.outputHead));
  
    // Specs bar
    var specsC = BLOCK_COLORS.specs;
    g.append("rect")
      .attr("x", pad.left)
      .attr("y", specsY)
      .attr("width", bw)
      .attr("height", specsH)
      .attr("rx", 8)
      .attr("fill", specsC.fill)
      .attr("stroke", specsC.stroke)
      .attr("stroke-width", 1);
    var specsItems = [
      arch.architectureType === "moe" ? "MoE" : "Dense",
      (arch.numLayers || "?") + " layers",
      arch.attentionType,
      arch.contextWindow ? formatContextWindow(arch.contextWindow) + " ctx" : null,
      isMoE ? (arch.numExperts || "?") + " experts" : null,
    ].filter(Boolean);
    var specSpacing = bw / (specsItems.length + 1);
    specsItems.forEach(function (txt, i) {
      g.append("text")
        .attr("x", pad.left + specSpacing * (i + 1))
        .attr("y", specsY + specsH / 2)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("fill", mutedFg)
        .attr("font-size", "11px")
        .attr("font-weight", 500)
        .attr("font-family", "inherit")
        .text(txt);
    });
  
    // Total weight memory (right side, next to specs bar)
    g.append("line")
      .attr("x1", memX)
      .attr("y1", specsY + 4)
      .attr("x2", memX + memLabelOffset - 16)
      .attr("y2", specsY + 4)
      .attr("stroke", borderColor)
      .attr("stroke-width", 1);
    g.append("text")
      .attr("x", memX)
      .attr("y", specsY + 22)
      .attr("text-anchor", "start")
      .attr("fill", fg)
      .attr("font-size", "14px")
      .attr("font-weight", 700)
      .attr("font-family", "'Montserrat', monospace")
      .text("Total: " + fmtBytes(mem.total));
    g.append("text")
      .attr("x", memX)
      .attr("y", specsY + 40)
      .attr("text-anchor", "start")
      .attr("fill", mutedFg)
      .attr("font-size", "10px")
      .attr("font-family", "inherit")
      .text(formatParamCount(arch.totalParams) + " \u00D7 " +
            (arch.ckptBytes || 2) + "B (" + (arch.ckptDtype || "bf16").toUpperCase() + ")");
  }
  
  // =========================================================================
  // Tab initialization & wiring
  // =========================================================================
  
  var selectedArch = null;
  var expandedBlocks = new Set();
  var memConfig = null;

  // Share state with topology module
  A.getState = function() { return { selectedArch: selectedArch, memConfig: memConfig }; };
  
  // Snap a value to the nearest option in a list
  function snapToOption(val, options) {
    var best = options[0];
    for (var i = 0; i < options.length; i++) {
      if (options[i] <= val) best = options[i];
    }
    return best;
  }
  
  var MML_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];
  
  function initMemConfig(arch) {
    var isMoE = arch.architectureType === "moe";
    // max-model-len defaults to the model's context window, snapped to nearest dropdown option
    var ctxWindow = arch.contextWindow || 8192;
    var defaultMml = snapToOption(ctxWindow, MML_OPTIONS);
    return {
      deviceId: "h200-141",
      dp: arch.defaultDp || 1,
      tp: arch.defaultTp || 8,
      epEnabled: isMoE,
      weightDtypeId: defaultWeightDtypeId(arch),
      kvDtypeId: defaultWeightDtypeId(arch) === "bf16" ? "bf16" : "fp8",
      // vLLM serve params (defaults match vllm CLI defaults / benchmark config)
      maxModelLen: defaultMml,
      maxNumBatchedTokens: 8192,   // vllm default with chunked prefill
      maxNumSeqs: 256,             // vllm default
      gpuMemUtil: 0.9,             // vllm default
    };
  }
  
  function selectModel(arch) {
    selectedArch = arch;
    expandedBlocks = new Set();
    memConfig = initMemConfig(arch);
  
    // Highlight selector card
    document.querySelectorAll(".arch-model-card").forEach(function (el) {
      el.classList.toggle("selected", el.dataset.archId === arch.id);
    });
  
    renderSelectedArch();
  }
  
  function renderSelectedArch() {
    var detailEl = document.getElementById("arch-detail");
    if (!detailEl || !selectedArch) return;
  
    var arch = selectedArch;
    var isMoE = arch.architectureType === "moe";
  
    // Header
    var header =
      '<div class="arch-detail-header">' +
      "<h3>" +
      escHtml(arch.name) +
      "</h3>" +
      '<div class="arch-badges">' +
      '<span class="arch-badge">' +
      (isMoE ? "MoE" : "Dense") +
      "</span>" +
      '<span class="arch-badge">' +
      (arch.attentionType === "AlternatingSinkGQA"
        ? "Sink/Full GQA"
        : arch.attentionType) +
      "</span>" +
      '<span class="arch-badge">' +
      formatParamCount(arch.totalParams) +
      "</span>" +
      (isMoE
        ? '<span class="arch-badge accent">' +
          formatParamCount(arch.activeParams) +
          " active</span>"
        : "") +
      "</div>" +
      (arch.developer
        ? '<p class="arch-meta">by ' +
          escHtml(arch.developer) +
          (arch.releaseDate
            ? " \u00B7 " +
              new Date(arch.releaseDate).toLocaleDateString("en-US", {
                year: "numeric",
                month: "short",
                day: "numeric",
              })
            : "") +
          (arch.sourceUrl
            ? ' \u00B7 <a href="' +
              arch.sourceUrl +
              '" target="_blank" rel="noopener">Source</a>'
            : "") +
          "</p>"
        : "") +
      "</div>";
  
    // Features
    var features = "";
    if (arch.features && arch.features.length) {
      features =
        '<div class="arch-features"><span class="arch-feat-label">Features:</span>' +
        arch.features
          .map(function (f) {
            return '<span class="arch-feat-badge">' + escHtml(f) + "</span>";
          })
          .join("") +
        "</div>";
    }
  
    detailEl.innerHTML =
      header +
      '<div class="arch-content-grid">' +
      '<div class="arch-diagram-col"><svg id="arch-svg"></svg></div>' +
      '<div class="arch-mem-col" id="arch-mem-panel"></div>' +
      "</div>" +
      features +
      '<div class="arch-topo-full" id="arch-topo-panel"></div>';
  
    // Render diagram
    var svgEl = document.getElementById("arch-svg");
    if (svgEl) {
      var onBlockClick = function (blockId) {
        if (expandedBlocks.has(blockId)) expandedBlocks.delete(blockId);
        else expandedBlocks.add(blockId);
        renderDiagram(svgEl, arch, expandedBlocks, onBlockClick);
      };
      renderDiagram(svgEl, arch, expandedBlocks, onBlockClick);
    }
  
    // Render memory controls + panel + topology
    renderMemoryControls();
    renderDeviceTopology();
  }
  
  function renderMemoryControls() {
    var panel = document.getElementById("arch-mem-panel");
    if (!panel || !selectedArch || !memConfig) return;
  
    var arch = selectedArch;
    var cfg = memConfig;
    var isMoE = arch.architectureType === "moe";
  
    // Controls
    var gpuOpts = GPU_DEVICES.map(function (d) {
      return (
        '<option value="' +
        d.id +
        '"' +
        (d.id === cfg.deviceId ? " selected" : "") +
        ">" +
        escHtml(d.label) +
        "</option>"
      );
    }).join("");
    var wOpts = WEIGHT_DTYPES.map(function (d) {
      return (
        '<option value="' +
        d.id +
        '"' +
        (d.id === cfg.weightDtypeId ? " selected" : "") +
        ">" +
        escHtml(d.label) +
        "</option>"
      );
    }).join("");
    var kvOpts = KV_DTYPES.map(function (d) {
      return (
        '<option value="' +
        d.id +
        '"' +
        (d.id === cfg.kvDtypeId ? " selected" : "") +
        ">" +
        escHtml(d.label) +
        "</option>"
      );
    }).join("");
  
    var controlsHtml =
      '<div class="arch-mem-controls">' +
      '<label class="arch-ctrl-full"><span>GPU</span><select id="mem-gpu">' +
      gpuOpts +
      "</select></label>" +
      '<div class="arch-ctrl-section">Parallelism</div>' +
      '<div class="arch-ctrl-row"><span>DP</span><div class="arch-stepper"><button data-field="dp" data-dir="-1">\u2212</button><span id="mem-dp-val">' +
      cfg.dp +
      '</span><button data-field="dp" data-dir="1">+</button></div></div>' +
      '<div class="arch-ctrl-row"><span>TP</span><div class="arch-stepper"><button data-field="tp" data-dir="-1">\u00F72</button><span id="mem-tp-val">' +
      cfg.tp +
      '</span><button data-field="tp" data-dir="1">\u00D72</button></div></div>' +
      '<div class="arch-ctrl-row"><span>EP' +
      (!isMoE ? " <small>(dense)</small>" : "") +
      '</span><button id="mem-ep-toggle" class="arch-ep-btn' +
      (cfg.epEnabled && isMoE ? " on" : "") +
      '"' +
      (!isMoE ? " disabled" : "") +
      ">" +
      (cfg.epEnabled && isMoE ? "ON" : "OFF") +
      "</button></div>" +
      '<div class="arch-ctrl-row"><span>Total GPUs</span><span class="arch-gpu-summary">DP' +
      cfg.dp +
      " \u00D7 TP" +
      cfg.tp +
      " = " +
      cfg.dp * cfg.tp +
      (cfg.epEnabled && isMoE ? " \u00B7 EP on" : "") +
      "</span></div>" +
      '<div class="arch-ctrl-section">Precision & KV</div>' +
      '<label><span>Weight dtype</span><select id="mem-wdtype">' +
      wOpts +
      "</select></label>" +
      '<label><span>KV dtype</span><select id="mem-kvdtype">' +
      kvOpts +
      "</select></label>" +
      '<div class="arch-ctrl-section">vLLM Serve Params</div>' +
      '<label><span>max-model-len</span><select id="mem-mml">' +
      [2048, 4096, 8192, 16384, 32768, 65536, 131072].map(function (v) {
        return '<option value="' + v + '"' + (v === cfg.maxModelLen ? ' selected' : '') + '>' +
          (v >= 1024 ? (v / 1024) + 'K' : v) + '</option>';
      }).join('') +
      '</select></label>' +
      '<label><span>max-num-batched-tokens</span><select id="mem-mbt">' +
      [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768].map(function (v) {
        return '<option value="' + v + '"' + (v === cfg.maxNumBatchedTokens ? ' selected' : '') + '>' +
          (v >= 1024 ? (v / 1024) + 'K' : v) + '</option>';
      }).join('') +
      '</select></label>' +
      '<label><span>max-num-seqs</span><select id="mem-mns">' +
      [32, 64, 128, 256, 512, 1024].map(function (v) {
        return '<option value="' + v + '"' + (v === cfg.maxNumSeqs ? ' selected' : '') + '>' + v + '</option>';
      }).join('') +
      '</select></label>' +
      '<label><span>gpu-memory-utilization</span><select id="mem-gmu">' +
      [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95].map(function (v) {
        return '<option value="' + v + '"' + (v === cfg.gpuMemUtil ? ' selected' : '') + '>' +
          (v * 100).toFixed(0) + '%</option>';
      }).join('') +
      '</select></label>' +
      "</div>";
  
    var breakdownContainer = '<div id="arch-mem-breakdown"></div>';
    panel.innerHTML = controlsHtml + breakdownContainer;
  
    // Wire up controls
    panel
      .querySelector("#mem-gpu")
      .addEventListener("change", function (e) {
        memConfig.deviceId = e.target.value;
        renderMemoryControls();
      });
    panel
      .querySelector("#mem-wdtype")
      .addEventListener("change", function (e) {
        memConfig.weightDtypeId = e.target.value;
        renderMemoryControls();
      });
    panel
      .querySelector("#mem-kvdtype")
      .addEventListener("change", function (e) {
        memConfig.kvDtypeId = e.target.value;
        renderMemoryControls();
      });
    panel.querySelector("#mem-mml").addEventListener("change", function (e) {
      memConfig.maxModelLen = Number(e.target.value);
      renderMemoryControls();
    });
    panel.querySelector("#mem-mbt").addEventListener("change", function (e) {
      memConfig.maxNumBatchedTokens = Number(e.target.value);
      renderMemoryControls();
    });
    panel.querySelector("#mem-mns").addEventListener("change", function (e) {
      memConfig.maxNumSeqs = Number(e.target.value);
      renderMemoryControls();
    });
    panel.querySelector("#mem-gmu").addEventListener("change", function (e) {
      memConfig.gpuMemUtil = Number(e.target.value);
      renderMemoryControls();
    });
  
    var epBtn = panel.querySelector("#mem-ep-toggle");
    if (epBtn && isMoE) {
      epBtn.addEventListener("click", function () {
        memConfig.epEnabled = !memConfig.epEnabled;
        renderMemoryControls();
      });
    }
  
    // Stepper buttons
    panel.querySelectorAll(".arch-stepper button").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var field = btn.dataset.field;
        var dir = Number(btn.dataset.dir);
        if (field === "tp") {
          // multiply/divide by 2
          memConfig.tp =
            dir > 0
              ? Math.min(64, memConfig.tp * 2)
              : Math.max(1, memConfig.tp / 2);
        } else {
          memConfig[field] = Math.max(
            1,
            Math.min(64, memConfig[field] + dir)
          );
        }
        renderMemoryControls();
      });
    });
  
    updateMemBreakdown();
    renderDeviceTopology();
  }
  
  function updateMemBreakdown() {
    var el = document.getElementById("arch-mem-breakdown");
    if (!el || !selectedArch || !memConfig) return;
    renderMemoryPanel(el, selectedArch, memConfig);
  }
  
  // =========================================================================
  // =========================================================================
  // Model selector grid
  // =========================================================================
  
  function renderModelCard(arch) {
    var isMoE = arch.architectureType === "moe";
    var paramStr = formatParamCount(arch.totalParams);
    var activeStr = isMoE
      ? formatParamCount(arch.activeParams) + " active"
      : "";
    return (
      '<div class="arch-model-card" data-arch-id="' +
      arch.id +
      '">' +
      '<div class="arch-card-top">' +
      '<h4 class="arch-card-name">' +
      escHtml(arch.name) +
      "</h4>" +
      '<div class="arch-card-top-badges">' +
      '<span class="arch-badge-sm">' +
      paramStr +
      "</span>" +
      (isMoE
        ? '<span class="arch-badge-moe">MoE</span>'
        : "") +
      "</div>" +
      "</div>" +
      '<div class="arch-card-tags">' +
      '<span class="arch-badge">' +
      (isMoE ? "MoE" : "Dense") +
      "</span>" +
      '<span class="arch-badge">' +
      arch.attentionType +
      "</span>" +
      (activeStr
        ? '<span class="arch-badge accent">' + activeStr + "</span>"
        : "") +
      "</div>" +
      '<div class="arch-card-info">' +
      escHtml(arch.developer || "") +
      (arch.contextWindow
        ? " \u00B7 " + formatContextWindow(arch.contextWindow) + " ctx"
        : "") +
      "</div>" +
      "</div>"
    );
  }
  
  function renderModelGrid() {
    var grid = document.getElementById("arch-model-grid");
    if (!grid) return;
  
    var moeModels = MODEL_ARCHITECTURES.filter(function (a) { return a.architectureType === "moe"; });
    var denseModels = MODEL_ARCHITECTURES.filter(function (a) { return a.architectureType !== "moe"; });
  
    var html =
      '<div class="arch-grid-section">' +
      '<div class="arch-grid-label">MoE Models</div>' +
      '<div class="arch-grid-row">' +
      moeModels.map(renderModelCard).join("") +
      '</div></div>' +
      '<div class="arch-grid-section">' +
      '<div class="arch-grid-label">Dense Models</div>' +
      '<div class="arch-grid-row">' +
      denseModels.map(renderModelCard).join("") +
      '</div></div>';
  
    grid.innerHTML = html;
  
    grid.querySelectorAll(".arch-model-card").forEach(function (card) {
      card.addEventListener("click", function () {
        var arch = MODEL_ARCHITECTURES.find(function (a) {
          return a.id === card.dataset.archId;
        });
        if (arch) selectModel(arch);
      });
    });
  }
  
  // =========================================================================
  // Public init — called by tabs.js when architecture tab is activated
  // =========================================================================
  
  var initialized = false;
  
  window.initModelArchitectures = function () {
    if (initialized) return;
    initialized = true;
  
    renderModelGrid();
    // Auto-select first model
    if (MODEL_ARCHITECTURES.length > 0) {
      selectModel(MODEL_ARCHITECTURES[0]);
    }
  };
})();
