/* ===========================================================================
 * MindLine 轨迹页（阶段三 · 方案 §4.4 / 决策 E4）
 *
 * 把「模型到底做了什么」画成一张可读的表：
 *   列 = 一次交互（同一个 trace_id），从左到右按时间排列；
 *   行 = 三个阶段：① 输入  ② 思考与回环  ③ 工具与输出。
 *
 * 数据来自 GET /api/trace?after=&session=（1 秒轮询增量）。轨迹是「log 的一部分」
 * （决策 E4）：内核既把它写进 ska.trace 日志，也落到 admin.db 的 traces 表，
 * 本页只负责把这份流水可视化，不参与任何决策。
 *
 * 页对象通过 window.EXTRA_PAGES 注册，由 app.js 末尾并入导航。
 * =========================================================================== */

'use strict';

(function () {
  const MAX_EVENTS = 800;     // 前端最多保留多少条轨迹（超出丢最旧的，避免无限增长）
  const MAX_COLS = 40;        // 最多画多少列（一次交互一列）

  const state = {
    timer: null,
    pollMs: 1000,
    events: [],               // 累积轨迹（按 seq 升序）
    bySeq: {},                // seq → 事件（点开看详情用）
    after: 0,                 // 增量游标
    lastSeq: 0,
    dirty: true,              // 有新数据 / 筛选变化 → 需要重绘泳道
  };

  window.EXTRA_PAGES = window.EXTRA_PAGES || [];

  window.EXTRA_PAGES.push({
    id: 'mindline',
    label: 'MindLine',
    desc: '输入 / 思考 / 工具与输出 三行泳道，逐列回放模型的一次交互',
    async load(root, c) {
      stopTimer();
      state.events = [];
      state.bySeq = {};
      state.after = 0;
      state.lastSeq = 0;
      state.dirty = true;

      // 顶部筛选的状态（本地过滤，不影响轮询：轮询始终取全量流）
      const filters = { session: '', tools: false, errors: false, range: 0 };

      renderShell();
      // 初次加载：先补一次历史（把 after 之前的也尽量拉一些），再进入轮询。
      await tick(true);
      startTimer();
      window.__teardown = stopTimer;

      // ---------------------------------------------------------------- 骨架
      function renderShell() {
        c.innerHTML = `
          <div class="ml-filters">
            <div class="ml-f">
              <label>会话</label>
              <select id="ml-session"><option value="">全部</option></select>
            </div>
            <label class="ml-cb"><input type="checkbox" id="ml-tools"> 只看工具</label>
            <label class="ml-cb"><input type="checkbox" id="ml-err"> 只看错误</label>
            <div class="ml-f">
              <label>时间范围</label>
              <select id="ml-range">
                <option value="0">全部</option>
                <option value="300">近 5 分钟</option>
                <option value="1800">近 30 分钟</option>
                <option value="3600">近 1 小时</option>
              </select>
            </div>
            <button class="ghost mini" id="ml-clear">清空轨迹</button>
            <span class="hint" id="ml-stat"></span>
          </div>
          <div class="ml-wrap">
            <div class="ml-labels">
              <div class="ml-lab head">交互</div>
              <div class="ml-lab lane-input">输入</div>
              <div class="ml-lab lane-think">思考 / 回环</div>
              <div class="ml-lab lane-act">工具 &amp; 输出</div>
            </div>
            <div class="ml-cols" id="ml-cols"></div>
          </div>
          <div class="panel" id="ml-detail">
            <p class="hint">点开任意一格查看完整详情。</p>
          </div>`;

        c.querySelector('#ml-session').onchange = (e) => { filters.session = e.target.value; state.dirty = true; renderGrid(); };
        c.querySelector('#ml-tools').onchange = (e) => { filters.tools = e.target.checked; state.dirty = true; renderGrid(); };
        c.querySelector('#ml-err').onchange = (e) => { filters.errors = e.target.checked; state.dirty = true; renderGrid(); };
        c.querySelector('#ml-range').onchange = (e) => { filters.range = parseInt(e.target.value, 10) || 0; state.dirty = true; renderGrid(); };
        c.querySelector('#ml-clear').onclick = async () => {
          if (!confirm('清空所有轨迹（内存缓冲 + 数据库）？')) return;
          try {
            await api('DELETE', '/api/trace');
            state.events = []; state.bySeq = {}; state.after = 0; state.lastSeq = 0;
            state.dirty = true; renderGrid();
            setMsg(root, '已清空轨迹');
          } catch (e) { setMsg(root, e.message, true); }
        };
      }

      // ---------------------------------------------------------------- 筛选
      function visible(events) {
        const cutoff = filters.range > 0 ? (Date.now() / 1000 - filters.range) : 0;
        return events.filter((e) => {
          if (filters.session && e.session !== filters.session) return false;
          if (cutoff && (e.ts || 0) < cutoff) return false;
          const kind = String(e.kind || '');
          if (filters.tools && !kind.startsWith('tool')) return false;
          if (filters.errors && !kind.includes('error')) return false;
          return true;
        });
      }

      function kindClass(kind) {
        const k = String(kind || '');
        if (k.includes('error')) return 'err';
        if (k.startsWith('tool')) return 'tool';
        return '';
      }

      // ---------------------------------------------------------------- 渲染
      function renderGrid() {
        const colsBox = c.querySelector('#ml-cols');
        if (!colsBox) return;
        // 会话下拉：从累积事件里取去重会话名，保留当前选中项。
        syncSessionOptions();
        state.dirty = false;

        const shown = visible(state.events);
        // 按 trace_id 归组（保持首次出现顺序）；一次交互 = 一列。
        const order = [];
        const groups = new Map();
        for (const e of shown) {
          const tid = e.trace_id || ('seq-' + e.seq);
          if (!groups.has(tid)) { groups.set(tid, { input: [], think: [], act: [] }); order.push(tid); }
          const g = groups.get(tid);
          (g[e.lane] || g.think).push(e);
        }
        const use = order.slice(-MAX_COLS);

        colsBox.innerHTML = use.map((tid) => {
          const g = groups.get(tid);
          const first = (g.input[0] || g.think[0] || g.act[0] || {});
          const sandbox = g.input.concat(g.think, g.act).some((x) => x.sandbox);
          const head = `${fmtTime(first.ts)}<small>${esc((first.session || '-').slice(0, 8))}</small>`
            + (sandbox ? '<span class="badge warn">沙盒</span>' : '');
          return `<div class="ml-col">
            <div class="ml-head">${head}</div>
            ${cell('input', g.input)}
            ${cell('think', g.think)}
            ${cell('act', g.act)}
          </div>`;
        }).join('') || '<p class="hint" style="padding:12px">（暂无轨迹：与智能体聊两句就有了）</p>';

        colsBox.querySelectorAll('.ml-chip').forEach((el) => {
          el.onclick = () => showDetail(parseInt(el.dataset.seq, 10));
        });
        // 横向滚到最右（最新一列），贴合「从左到右按时间」的阅读方向。
        colsBox.scrollLeft = colsBox.scrollWidth;
        updateStat();
      }

      function cell(lane, list) {
        if (!list.length) return `<div class="ml-cell lane-${lane} empty"></div>`;
        const chips = list.map((e) => {
          const tk = e.tokens ? ` <span class="ml-tk">${e.tokens}t</span>` : '';
          const title = e.title || e.kind;
          return `<div class="ml-chip ${kindClass(e.kind)}" data-seq="${e.seq}"
                    title="${esc(e.kind)}">${esc(title)}${tk}</div>`;
        }).join('');
        return `<div class="ml-cell lane-${lane}">${chips}</div>`;
      }

      function syncSessionOptions() {
        const sel = c.querySelector('#ml-session');
        if (!sel) return;
        const names = Array.from(new Set(state.events.map((e) => e.session).filter(Boolean))).sort();
        const sig = names.join('|');
        if (sel.dataset.sig === sig) return;
        sel.dataset.sig = sig;
        sel.innerHTML = '<option value="">全部</option>'
          + names.map((n) => `<option ${n === filters.session ? 'selected' : ''}>${esc(n)}</option>`).join('');
      }

      function updateStat() {
        const box = c.querySelector('#ml-stat');
        if (!box) return;
        const shown = visible(state.events).length;
        box.textContent = `已载 ${state.events.length} 条 · 显示 ${shown} 条 · 游标 seq=${state.lastSeq}`;
      }

      function showDetail(seq) {
        const box = c.querySelector('#ml-detail');
        const e = state.bySeq[seq];
        if (!box || !e) return;
        box.innerHTML = `
          <h3>#${e.seq} · ${esc(e.lane)}/${esc(e.kind)}</h3>
          <div class="hint">会话 ${esc(e.session || '-')} · ${fmtTime(e.ts)}
            · trace ${esc(e.trace_id || '-')} · tokens ${e.tokens || 0}
            ${e.sandbox ? ' · <span class="badge warn">沙盒</span>' : ''}</div>
          <div><b>${esc(e.title || '')}</b></div>
          <pre class="out">${esc(e.detail || '（无详情）')}</pre>`;
      }

      // ---------------------------------------------------------------- 轮询
      function startTimer() {
        if (state.timer) return;
        state.timer = setInterval(tick, state.pollMs);
      }
      function stopTimer() {
        if (state.timer) { clearInterval(state.timer); state.timer = null; }
      }

      async function tick() {
        if (!c.querySelector('#ml-cols')) { stopTimer(); return; }
        try {
          const r = await api('GET', `/api/trace?after=${state.after}&limit=${MAX_EVENTS}`);
          if (r.events && r.events.length) {
            for (const e of r.events) { state.events.push(e); state.bySeq[e.seq] = e; }
            // 只保留最近 MAX_EVENTS 条，丢弃最旧的（同时清索引，避免内存泄漏）。
            if (state.events.length > MAX_EVENTS) {
              const drop = state.events.splice(0, state.events.length - MAX_EVENTS);
              for (const e of drop) delete state.bySeq[e.seq];
            }
            state.dirty = true;
          }
          state.after = r.last_seq;
          state.lastSeq = r.last_seq;
          if (state.dirty) renderGrid(); else updateStat();
        } catch (e) {
          stopTimer();
          setMsg(root, e.message, true);
        }
      }
    },
  });
})();
