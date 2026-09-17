/**
 * PseudoFlow Web UI Engine
 * Path A: Neuro-Symbolic Translation & Visual Compiler
 */

// Initialize Mermaid.js with Dark Theme
mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
  flowchart: {
    curve: 'basis',
    nodeSpacing: 50,
    rankSpacing: 50,
    useMaxWidth: true
  }
});

/* ==============================================================================
   Presets Data
   ============================================================================== */
const PRESETS = {
  even_odd: `START
INPUT n
IF n % 2 == 0
    OUTPUT "Even"
ELSE
    OUTPUT "Odd"
END IF
END`,

  nested_for: `START
INPUT limit
FOR i = 1 TO limit
    FOR j = 1 TO i
        LET product = i * j
        OUTPUT product
    END FOR
END FOR
END`,

  while_sum: `START
INPUT max_count
LET sum = 0
LET i = 1
WHILE i <= max_count
    LET sum = sum + i
    LET i = i + 1
END WHILE
OUTPUT sum
END`,

  linear_calc: `START
INPUT radius
LET pi = 3.14159
LET area = pi * radius * radius
LET circumference = 2 * pi * radius
OUTPUT area
OUTPUT circumference
END`,

  custom: `START
INPUT score
IF score >= 50
    OUTPUT "Passed"
ELSE
    OUTPUT "Failed"
END IF
END`
};

/* ==============================================================================
   Statement Role Classification (Corpus / Tagged Model Alignment)
   ============================================================================== */
const TAG_DEFINITIONS = {
  'start-end': { label: 'START/END', color: '#8b5cf6', hlClass: 'hl-start-end' },
  'if':        { label: 'IF_DECISION', color: '#f59e0b', hlClass: 'hl-if' },
  'else':      { label: 'ELSE_BRANCH', color: '#ec4899', hlClass: 'hl-else' },
  'loop':      { label: 'LOOP_HEADER', color: '#3b82f6', hlClass: 'hl-loop' },
  'io':        { label: 'IO_NODE', color: '#10b981', hlClass: 'hl-io' },
  'process':   { label: 'PROCESS_NODE', color: '#f97316', hlClass: 'hl-process' },
  'end-block': { label: 'END_BLOCK', color: '#64748b', hlClass: 'hl-end-block' },
  'default':   { label: 'NONE', color: 'transparent', hlClass: '' }
};

/**
 * Classifies a single line of pseudocode into a role tag.
 * Mirrors the BiRNN statement classifier logic trained on Tagged_Corpus.json.
 */
function classifyLine(line) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith('//') || trimmed.startsWith('#')) {
    return 'default';
  }

  // START / END
  if (/^START\b/i.test(trimmed)) return 'start-end';
  if (/^END\b/i.test(trimmed) && !/^(END\s+IF|END\s+FOR|END\s+WHILE)\b/i.test(trimmed)) return 'start-end';

  // CONTROL FLOW
  if (/^IF\b/i.test(trimmed)) return 'if';
  if (/^ELSE\b/i.test(trimmed)) return 'else';
  if (/^END\s+IF\b/i.test(trimmed)) return 'end-block';

  // LOOPS
  if (/^(FOR|WHILE)\b/i.test(trimmed)) return 'loop';
  if (/^(END\s+FOR|END\s+WHILE)\b/i.test(trimmed)) return 'end-block';

  // I/O
  if (/^(INPUT|OUTPUT|PRINT|READ|WRITE)\b/i.test(trimmed)) return 'io';

  // PROCESS / ASSIGNMENT
  if (/^(LET\b|[a-zA-Z_]\w*\s*=)/i.test(trimmed)) return 'process';

  return 'process';
}

/**
 * Returns the short indicator string (e.g. '(s)', '(f)', '(i)') for a given line.
 */
function getLineIndicator(line, role) {
  const trimmed = line.trim();
  if (!trimmed || role === 'default') return '';

  if (role === 'start-end') {
    if (/^START\b/i.test(trimmed)) return '(s)';
    if (/^END\b/i.test(trimmed)) return '(e)';
    return '(s)';
  }
  if (role === 'if') return '(i)';
  if (role === 'else') return '(el)';
  if (role === 'loop') {
    if (/^FOR\b/i.test(trimmed)) return '(f)';
    if (/^WHILE\b/i.test(trimmed)) return '(w)';
    return '(f)';
  }
  if (role === 'io') {
    if (/^(INPUT|READ)\b/i.test(trimmed)) return '(in)';
    if (/^(OUTPUT|PRINT|WRITE)\b/i.test(trimmed)) return '(out)';
    return '(io)';
  }
  if (role === 'process') return '(p)';
  if (role === 'end-block') {
    if (/^END\s+IF\b/i.test(trimmed)) return '(eb)';
    if (/^END\s+FOR\b/i.test(trimmed)) return '(eb)';
    if (/^END\s+WHILE\b/i.test(trimmed)) return '(eb)';
    return '(eb)';
  }
  return '';
}

/* ==============================================================================
   Deterministic Flowchart Graph Assembler (Path A Engine)
   Produces 100% valid Mermaid script with correct node shapes & logic wiring
   ============================================================================== */
class FlowchartAssembler {
  constructor() {
    this.nodes = [];
    this.edges = [];
    this.nodeCount = 0;
  }

  sanitize(text) {
    return text
      .replace(/"/g, '#quot;')
      .replace(/\[/g, '#91;')
      .replace(/\]/g, '#93;')
      .replace(/\{/g, '#123;')
      .replace(/\}/g, '#125;')
      .replace(/\(/g, '#40;')
      .replace(/\)/g, '#41;');
  }

  createNode(text, role) {
    this.nodeCount++;
    const id = `node${this.nodeCount}`;
    const safeText = this.sanitize(text.trim());
    let shape = '';

    // Notice: NO SPACE between id and opening bracket for strict parser compliance!
    switch (role) {
      case 'start-end':
        shape = `${id}([" ${safeText} "])`;
        break;
      case 'io':
        shape = `${id}[/" ${safeText} "/]`;
        break;
      case 'if':
      case 'loop':
        shape = `${id}{" ${safeText} "}`;
        break;
      case 'process':
      default:
        shape = `${id}[" ${safeText} "]`;
        break;
    }

    this.nodes.push({ id, shape, role, text: text.trim() });
    return id;
  }

  assemble(pseudocode) {
    const rawLines = pseudocode.split('\n');
    const statements = [];

    // Parse non-empty lines
    for (let i = 0; i < rawLines.length; i++) {
      const line = rawLines[i].trim();
      if (line) {
        statements.push({
          raw: line,
          role: classifyLine(line),
          lineNum: i + 1
        });
      }
    }

    if (statements.length === 0) {
      return { script: 'flowchart TD\n    empty[" Empty Pseudocode "]', nodeCount: 0, edgeCount: 0 };
    }

    const blockStack = [];
    let pendingPredecessors = [];

    for (let i = 0; i < statements.length; i++) {
      const stmt = statements[i];

      if (stmt.role === 'else') {
        // Find matching IF on stack
        const topBlock = blockStack[blockStack.length - 1];
        if (topBlock && topBlock.type === 'IF') {
          topBlock.thenLeaves = [...pendingPredecessors];
          topBlock.inElse = true;
          // Predecessor for first node inside ELSE branch is the IF node itself
          pendingPredecessors = [{ id: topBlock.id, branch: 'No' }];
        }
        continue;
      }

      if (stmt.role === 'end-block') {
        if (/^END\s+IF/i.test(stmt.raw)) {
          const topBlock = blockStack.pop();
          if (topBlock && topBlock.type === 'IF') {
            if (topBlock.inElse) {
              topBlock.elseLeaves = [...pendingPredecessors];
              pendingPredecessors = [...topBlock.thenLeaves, ...topBlock.elseLeaves];
            } else {
              // No ELSE branch was provided; then leaves + IF node directly (No)
              topBlock.thenLeaves = [...pendingPredecessors];
              pendingPredecessors = [...topBlock.thenLeaves, { id: topBlock.id, branch: 'No' }];
            }
          }
        } else if (/^(END\s+FOR|END\s+WHILE)/i.test(stmt.raw)) {
          const topBlock = blockStack.pop();
          if (topBlock && (topBlock.type === 'FOR' || topBlock.type === 'WHILE')) {
            // Loop body leaves wire back to the loop header
            for (const pred of pendingPredecessors) {
              this.edges.push(`    ${pred.id} --> ${topBlock.id}`);
            }
            // Next node connects from loop header exit
            pendingPredecessors = [{ id: topBlock.id, branch: 'No' }];
          }
        }
        continue;
      }

      // Create new graph node
      const currentId = this.createNode(stmt.raw, stmt.role);

      // Connect pending predecessors
      for (const pred of pendingPredecessors) {
        if (pred.branch) {
          this.edges.push(`    ${pred.id} -- ${pred.branch} --> ${currentId}`);
        } else {
          this.edges.push(`    ${pred.id} --> ${currentId}`);
        }
      }

      // Handle block starters
      if (stmt.role === 'if') {
        blockStack.push({
          type: 'IF',
          id: currentId,
          thenLeaves: [],
          elseLeaves: [],
          inElse: false
        });
        pendingPredecessors = [{ id: currentId, branch: 'Yes' }];
      } else if (stmt.role === 'loop') {
        const loopType = /^FOR/i.test(stmt.raw) ? 'FOR' : 'WHILE';
        blockStack.push({
          type: loopType,
          id: currentId
        });
        pendingPredecessors = [{ id: currentId, branch: 'Yes' }];
      } else {
        pendingPredecessors = [{ id: currentId, branch: null }];
      }
    }

    // Build final Mermaid script
    let script = 'flowchart TD\n';
    for (const node of this.nodes) {
      script += `    ${node.shape}\n`;
    }
    for (const edge of this.edges) {
      script += `${edge}\n`;
    }

    return {
      script,
      nodeCount: this.nodes.length,
      edgeCount: this.edges.length
    };
  }
}

/* ==============================================================================
   UI Manager & Event Handlers
   ============================================================================== */
class UIManager {
  constructor() {
    this.textarea = document.getElementById('editor-textarea');
    this.highlight = document.getElementById('editor-highlight');
    this.gutter = document.getElementById('line-gutter');
    this.runBtn = document.getElementById('run-btn');
    this.presetSelect = document.getElementById('preset-select');
    this.mermaidOutput = document.getElementById('mermaid-output');
    this.rawCodeView = document.getElementById('raw-code-view');
    this.sizeIndicator = document.getElementById('size-indicator');
    this.statusText = document.getElementById('status-text');
    this.statusNodes = document.getElementById('status-nodes');
    this.lineBadge = document.getElementById('line-count-badge');
    this.charBadge = document.getElementById('char-count-badge');
    this.paneGutter = document.getElementById('pane-gutter');
    this.leftPane = document.getElementById('left-pane');
    this.rightPane = document.getElementById('right-pane');
    this.canvasContainer = document.getElementById('canvas-container');
    this.rightGutter = document.getElementById('right-gutter');

    // View state
    this.currentView = 'diagram';
    this.zoomScale = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.startX = 0;
    this.startY = 0;
    this.currentMermaidCode = '';

    this.initEvents();
    this.updateEditor();
    this.updateSizeIndicator();
    this.compileFlowchart();
  }

  initEvents() {
    // Editor sync and typing
    this.textarea.addEventListener('input', () => {
      this.updateEditor();
    });

    this.textarea.addEventListener('scroll', () => {
      this.syncScroll();
    });

    // Legend badge click to pulse highlight lines of that role
    document.querySelectorAll('.legend-badge[data-role]').forEach(badge => {
      badge.addEventListener('click', () => {
        const role = badge.getAttribute('data-role');
        this.pulseRole(role, badge);
      });
    });

    // Right indicator badge click to pulse that role
    if (this.rightGutter) {
      this.rightGutter.addEventListener('click', (e) => {
        const badge = e.target.closest('.role-indicator-badge');
        if (badge) {
          const role = badge.getAttribute('data-role');
          const legendBadge = document.querySelector(`.legend-badge[data-role="${role}"]`);
          this.pulseRole(role, legendBadge);
        }
      });
    }

    // Keyboard shortcut (Ctrl+Enter to compile)
    this.textarea.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        this.compileFlowchart();
      }
      // Tab key indentation support
      if (e.key === 'Tab') {
        e.preventDefault();
        const start = this.textarea.selectionStart;
        const end = this.textarea.selectionEnd;
        this.textarea.value = this.textarea.value.substring(0, start) + '    ' + this.textarea.value.substring(end);
        this.textarea.selectionStart = this.textarea.selectionEnd = start + 4;
        this.updateEditor();
      }
    });

    // Run button
    this.runBtn.addEventListener('click', () => {
      this.compileFlowchart();
    });

    // Preset selector
    this.presetSelect.addEventListener('change', (e) => {
      const key = e.target.value;
      if (PRESETS[key]) {
        this.textarea.value = PRESETS[key];
        this.updateEditor();
        this.compileFlowchart();
      }
    });

    // View tabs (Diagram vs Source)
    document.getElementById('tab-diagram').addEventListener('click', () => {
      this.switchView('diagram');
    });
    document.getElementById('tab-code').addEventListener('click', () => {
      this.switchView('code');
    });

    // Zoom and Pan buttons
    document.getElementById('btn-zoom-in').addEventListener('click', () => this.zoom(0.15));
    document.getElementById('btn-zoom-out').addEventListener('click', () => this.zoom(-0.15));
    document.getElementById('btn-zoom-reset').addEventListener('click', () => this.resetTransform());

    // Copy buttons
    document.getElementById('btn-copy-mermaid').addEventListener('click', () => this.copyMermaid());
    document.getElementById('btn-copy-svg').addEventListener('click', () => this.copySVG());
    document.getElementById('export-btn').addEventListener('click', () => this.exportSVG());

    // Canvas panning with mouse
    this.canvasContainer.addEventListener('mousedown', (e) => {
      if (e.button === 0 && this.currentView === 'diagram') {
        this.isDragging = true;
        this.startX = e.clientX - this.panX;
        this.startY = e.clientY - this.panY;
        this.canvasContainer.style.cursor = 'grabbing';
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isDragging) {
        this.panX = e.clientX - this.startX;
        this.panY = e.clientY - this.startY;
        this.applyTransform();
      }
    });

    window.addEventListener('mouseup', () => {
      if (this.isDragging) {
        this.isDragging = false;
        this.canvasContainer.style.cursor = 'default';
      }
    });

    // Mouse wheel zoom
    this.canvasContainer.addEventListener('wheel', (e) => {
      if (this.currentView === 'diagram') {
        e.preventDefault();
        const delta = e.deltaY < 0 ? 0.1 : -0.1;
        this.zoom(delta);
      }
    }, { passive: false });

    // Window resize
    window.addEventListener('resize', () => {
      this.updateSizeIndicator();
    });

    // Resizable Divider
    this.initGutterDrag();

    // Theme toggle button
    document.getElementById('btn-theme').addEventListener('click', () => {
      document.body.classList.toggle('dark-editor');
    });

    // Layout split toggle button
    document.getElementById('btn-orientation').addEventListener('click', () => {
      const workspace = document.getElementById('workspace');
      workspace.classList.toggle('horizontal-split');
      this.updateSizeIndicator();
    });
  }

  /**
   * Synchronizes line classification dots, background tints, and rightmost badges with editor textarea.
   */
  updateEditor() {
    const text = this.textarea.value;
    const lines = text.split('\n');

    // Update metrics
    this.lineBadge.textContent = `${lines.length} lines`;
    this.charBadge.textContent = `${text.length} chars`;

    // Rebuild gutter, highlight overlay, and right indicator gutter
    let gutterHTML = '';
    let highlightHTML = '';
    let rightGutterHTML = '';

    lines.forEach((line, index) => {
      const role = classifyLine(line);
      const tagInfo = TAG_DEFINITIONS[role] || TAG_DEFINITIONS.default;
      const indicator = getLineIndicator(line, role);

      // Left gutter item with line number and colored dot
      gutterHTML += `
        <div class="gutter-row" data-role="${role}">
          <span class="role-dot" style="background-color: ${tagInfo.color};" data-role="${role}" title="${tagInfo.label}"></span>
          <span>${index + 1}</span>
        </div>`;

      // Highlight line underlay
      highlightHTML += `<div class="code-line ${tagInfo.hlClass}" data-role="${role}"></div>`;

      // Rightmost indicator badge (e.g. (s), (f), (i), etc.)
      rightGutterHTML += `
        <div class="right-gutter-row" data-role="${role}">
          ${indicator ? `<span class="role-indicator-badge" data-role="${role}" style="color: ${tagInfo.color}; background-color: ${tagInfo.color}1c; border: 1px solid ${tagInfo.color}55;" title="Click to highlight all ${tagInfo.label} statements">${indicator}</span>` : ''}
        </div>`;
    });

    this.gutter.innerHTML = gutterHTML;
    this.highlight.innerHTML = highlightHTML;
    if (this.rightGutter) {
      this.rightGutter.innerHTML = rightGutterHTML;
    }

    this.syncScroll();
  }

  syncScroll() {
    this.highlight.scrollTop = this.textarea.scrollTop;
    this.highlight.scrollLeft = this.textarea.scrollLeft;
    this.gutter.scrollTop = this.textarea.scrollTop;
    if (this.rightGutter) {
      this.rightGutter.scrollTop = this.textarea.scrollTop;
    }
  }

  /**
   * Pulses all lines, right indicators, and gutter dots matching the given role
   * with a single pulse of that role's color.
   */
  pulseRole(role, badgeElement) {
    const tagInfo = TAG_DEFINITIONS[role];
    if (!tagInfo) return;

    // Visual bounce on clicked legend badge
    if (badgeElement) {
      badgeElement.classList.remove('legend-pulsing');
      void badgeElement.offsetWidth;
      badgeElement.classList.add('legend-pulsing');
    }

    // Find all matching elements in DOM
    const codeLines = this.highlight.querySelectorAll(`.code-line[data-role="${role}"]`);
    const rightBadges = this.rightGutter ? this.rightGutter.querySelectorAll(`.right-gutter-row[data-role="${role}"] .role-indicator-badge`) : [];
    const gutterDots = this.gutter.querySelectorAll(`.gutter-row[data-role="${role}"] .role-dot`);

    const count = codeLines.length;

    // Trigger pulse on code lines
    codeLines.forEach(line => {
      line.style.setProperty('--pulse-color', tagInfo.color);
      line.style.setProperty('--pulse-bg-start', `${tagInfo.color}14`);
      line.style.setProperty('--pulse-bg-peak', `${tagInfo.color}66`);
      line.classList.remove('line-pulsing');
      void line.offsetWidth;
      line.classList.add('line-pulsing');
    });

    // Trigger pulse on right gutter badges
    rightBadges.forEach(badge => {
      badge.style.setProperty('--pulse-color', tagInfo.color);
      badge.classList.remove('badge-pulsing');
      void badge.offsetWidth;
      badge.classList.add('badge-pulsing');
    });

    // Trigger pulse on left gutter dots
    gutterDots.forEach(dot => {
      dot.style.setProperty('--pulse-color', tagInfo.color);
      dot.classList.remove('dot-pulsing');
      void dot.offsetWidth;
      dot.classList.add('dot-pulsing');
    });

    if (count > 0) {
      this.flashStatus(`Pulsed ${count} "${tagInfo.label}" statement(s)`);
    } else {
      this.flashStatus(`No "${tagInfo.label}" statements in code`);
    }
  }

  /**
   * Compiles the pseudocode using the deterministic Path A FlowchartAssembler
   * and renders the diagram via Mermaid.js.
   */
  async compileFlowchart() {
    const pseudocode = this.textarea.value;
    this.statusText.textContent = 'Compiling...';

    const assembler = new FlowchartAssembler();
    const result = assembler.assemble(pseudocode);
    this.currentMermaidCode = result.script;

    // Update raw script tab
    this.rawCodeView.textContent = result.script;

    // Render diagram
    const container = this.mermaidOutput;
    const graphId = 'flowchart-svg-' + Date.now();

    try {
      const { svg } = await mermaid.render(graphId, result.script);
      container.innerHTML = svg;

      // Ensure SVG is responsive and crisp
      const svgElement = container.querySelector('svg');
      if (svgElement) {
        svgElement.style.maxWidth = '100%';
        svgElement.style.height = 'auto';
      }

      this.statusText.textContent = 'Compiled Successfully';
      this.statusNodes.textContent = `${result.nodeCount} nodes • ${result.edgeCount} edges`;
      this.resetTransform();
    } catch (err) {
      console.error('Mermaid render error:', err);
      container.innerHTML = `
        <div style="color: #ef4444; font-family: monospace; padding: 20px; text-align: center;">
          <p style="font-size: 16px; font-weight: bold; margin-bottom: 8px;">Compilation Error</p>
          <p style="font-size: 13px; color: #9ca3af;">${err.message || 'Syntax error in generated graph.'}</p>
        </div>`;
      this.statusText.textContent = 'Error';
    }
  }

  switchView(view) {
    this.currentView = view;
    document.getElementById('tab-diagram').classList.toggle('active', view === 'diagram');
    document.getElementById('tab-code').classList.toggle('active', view === 'code');

    if (view === 'diagram') {
      this.mermaidOutput.style.display = 'flex';
      this.rawCodeView.style.display = 'none';
    } else {
      this.mermaidOutput.style.display = 'none';
      this.rawCodeView.style.display = 'block';
    }
  }

  zoom(factor) {
    this.zoomScale = Math.max(0.3, Math.min(3.0, this.zoomScale + factor));
    this.applyTransform();
  }

  resetTransform() {
    this.zoomScale = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.applyTransform();
  }

  applyTransform() {
    const svg = this.mermaidOutput.querySelector('svg');
    if (svg) {
      svg.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoomScale})`;
      svg.style.transition = this.isDragging ? 'none' : 'transform 0.1s ease-out';
    }
  }

  updateSizeIndicator() {
    const rect = this.rightPane.getBoundingClientRect();
    this.sizeIndicator.textContent = `Result Size: ${Math.round(rect.width)} x ${Math.round(rect.height)}`;
  }

  initGutterDrag() {
    let isResizing = false;

    this.paneGutter.addEventListener('mousedown', (e) => {
      isResizing = true;
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    });

    window.addEventListener('mousemove', (e) => {
      if (!isResizing) return;
      const containerWidth = document.getElementById('workspace').offsetWidth;
      const leftWidth = e.clientX;
      const percentage = (leftWidth / containerWidth) * 100;

      if (percentage > 15 && percentage < 85) {
        this.leftPane.style.flex = `0 0 ${percentage}%`;
        this.rightPane.style.flex = `0 0 ${100 - percentage}%`;
        this.updateSizeIndicator();
      }
    });

    window.addEventListener('mouseup', () => {
      if (isResizing) {
        isResizing = false;
        document.body.style.cursor = 'default';
        document.body.style.userSelect = 'auto';
      }
    });
  }

  async copyMermaid() {
    if (!this.currentMermaidCode) return;
    await navigator.clipboard.writeText(this.currentMermaidCode);
    this.flashStatus('Mermaid Code Copied!');
  }

  async copySVG() {
    const svg = this.mermaidOutput.querySelector('svg');
    if (!svg) return;
    await navigator.clipboard.writeText(svg.outerHTML);
    this.flashStatus('SVG Code Copied!');
  }

  exportSVG() {
    const svg = this.mermaidOutput.querySelector('svg');
    if (!svg) return;
    const blob = new Blob([svg.outerHTML], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pseudoflow_chart.svg';
    a.click();
    URL.revokeObjectURL(url);
    this.flashStatus('SVG Exported!');
  }

  flashStatus(msg) {
    const original = this.statusText.textContent;
    this.statusText.textContent = msg;
    setTimeout(() => {
      this.statusText.textContent = original;
    }, 2000);
  }
}

// Boot application once DOM is ready
window.addEventListener('DOMContentLoaded', () => {
  window.pseudoFlowApp = new UIManager();
});
