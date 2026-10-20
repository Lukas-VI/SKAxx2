/* ===========================================================================
 * 在线聊天页（阶段三 · 方案 §2.4 / 决策 E1、E2、E3）
 *
 * 这一页只负责「浏览器 ↔ 核心」的搬运与展示，真正的推理仍由心跳主循环中的 Mind 完成：
 *   发消息：POST /api/chat/send  → ChatHub.submit() → 总线 → Mind 五步（权限/记忆/追踪全白送）
 *   收回复：GET  /api/chat/poll  → ChatHub 出站队列（1 秒轮询，决策 E2）
 *
 * 多开（决策 E1）：每个会话是独立的上下文窗口，但共享长期记忆 / 画像 / 权限；
 *   会话可勾选「沙盒」（persist=false）——照常推理，但不写任何记忆库，适合边聊边调试、教智能体。
 *
 * 页对象通过 window.EXTRA_PAGES 注册，由 app.js 末尾并入导航（见 app.js 注释）。
 * 会用到的工具函数（api/esc/fmtTime/setMsg）都来自 app.js。
 * =========================================================================== */

'use strict';

(function () {
  // 本页的可变状态：只在打开期间存活；切页时由 window.__teardown 停止轮询。
  const state = {
    timer: null,        // 轮询定时器 id
    pollMs: 1000,       // 轮询间隔（来自 /api/chat/meta，默认 1 秒）
    meta: null,         // /api/chat/meta 的返回（身份/等级/默认落库/窗口）
    sessions: [],       // 会话列表缓存（渲染标签栏用）
    activeId: '',       // 当前打开的会话 id
    after: 0,           // 增量游标：只取 seq > after 的新消息
    signature: '',      // 会话列表指纹：无变化时跳过标签栏重绘
  };

  window.EXTRA_PAGES = window.EXTRA_PAGES || [];

  window.EXTRA_PAGES.push({
    id: 'chat',
    label: '在线聊天',
    desc: '多开会话 / 沙盒调试 / 共享长期记忆 — 与智能体实时对话',
    async load(root, c) {
      stopTimer();                      // 万一切页清理漏了，这里兜底
      state.sessions = [];
      state.activeId = '';
      state.after = 0;
      state.signature = '';

      // 1) 元信息：身份 / 等级 / 默认落库 / 轮询间隔 / 模型窗口
      const meta = await api('GET', '/api/chat/meta');
      state.meta = meta;
      state.pollMs = Math.max(parseInt(meta.poll_ms, 10) || 1000, 200);

      // 2) 现有会话列表
      const list = await api('GET', '/api/chat/sessions');
      state.sessions = list.sessions || [];

      render();
      // 已有会话就打开最近的一个；否则自动开一个默认会话，省得空白无从下手。
      if (state.sessions.length) await selectSession(state.sessions[0].id);
      else await createSession(meta.persist);

      // 注册清理钩子：切页时停掉 1 秒轮询（app.js 的 show() 会调用它）。
      window.__teardown = stopTimer;

      // ---------------------------------------------------------------- 骨架
      function render() {
        c.innerHTML = `
          <div class="chat-wrap">
            <div class="chat-tabs" id="ch-tabs"></div>
            <div class="chat-body">
              <div class="chat-status" id="ch-status"></div>
              <div class="chat-log" id="ch-log"></div>
              <div class="chat-input">
                <textarea id="ch-text" rows="2"
                  placeholder="输入消息，Enter 发送 / Shift+Enter 换行"></textarea>
                <button id="ch-send">发送</button>
              </div>
            </div>
          </div>`;
        c.querySelector('#ch-send').onclick = onSend;
        const ta = c.querySelector('#ch-text');
        ta.onkeydown = (ev) => {
          if (ev.key === 'Enter' && !ev.shiftKey) { ev.preventDefault(); onSend(); }
        };
        renderTabs();
        renderStatus();
      }

      // ---------------------------------------------------------------- 标签栏
      function renderTabs() {
        const box = c.querySelector('#ch-tabs');
        if (!box) return;
        // 指纹 = 当前选中 + 每个会话的 id/标题/落库/待回复；不变则不重绘，避免每 tick 闪动。
        const sig = JSON.stringify([state.activeId,
          state.sessions.map((s) => [s.id, s.title, s.persist, s.pending])]);
        if (sig === state.signature) return;
        state.signature = sig;

        const tabs = state.sessions.map((s) => {
          const cls = 'chat-tab' + (s.id === state.activeId ? ' active' : '');
          const marks = (s.persist ? '' : '<span class="badge warn">沙盒</span>')
            + (s.pending ? '<span class="badge">…</span>' : '');
          return `<div class="${cls}" data-id="${esc(s.id)}">${esc(s.title)} ${marks}</div>`;
        }).join('');
        box.innerHTML = tabs
          + '<div class="chat-tab ghost" id="ch-new">＋ 新会话</div>'
          + '<div class="chat-tab ghost" id="ch-new-sandbox">＋ 临时会话</div>'
          + '<div class="chat-tab ghost" id="ch-close">关闭当前</div>';

        box.querySelectorAll('.chat-tab[data-id]').forEach((el) => {
          el.onclick = () => selectSession(el.dataset.id);
        });
        box.querySelector('#ch-new').onclick = () => createSession(true);
        box.querySelector('#ch-new-sandbox').onclick = () => createSession(false);
        box.querySelector('#ch-close').onclick = closeSession;
      }

      // ---------------------------------------------------------------- 状态条
      function renderStatus() {
        const box = c.querySelector('#ch-status');
        if (!box) return;
        const cur = state.sessions.find((s) => s.id === state.activeId) || {};
        const m = state.meta || {};
        const win = (m.context_window > 0) ? m.context_window : '跟随提供商';
        box.innerHTML = `
          <span class="badge">身份 ${esc(m.identity || 'owner')}</span>
          <span class="badge">等级 ${esc(m.level || 'member')}</span>
          <span class="badge ${cur.persist === false ? 'warn' : 'ok'}">
            ${cur.persist === false ? '沙盒（不落库）' : '保存记忆'}</span>
          <span class="badge">上下文 ${cur.tokens || 0} / ${esc(String(win))}</span>
          ${cur.pending ? '<span class="badge">思考中…</span>' : ''}`;
      }

      // ---------------------------------------------------------------- 消息流
      function bubble(m) {
        const mine = m.role === 'user';
        return `<div class="bubble ${mine ? 'me' : 'bot'}">
          <div class="who">${mine ? '你' : 'SKA'} · ${fmtTime(m.ts || 0)}</div>
          <div class="txt">${esc(m.content)}</div></div>`;
      }

      function appendMessages(list) {
        const box = c.querySelector('#ch-log');
        if (!box || !list || !list.length) return;
        const frag = document.createElement('div');
        frag.innerHTML = list.map(bubble).join('');
        while (frag.firstChild) box.appendChild(frag.firstChild);
        box.scrollTop = box.scrollHeight;      // 新消息滚到底
      }

      // ---------------------------------------------------------------- 会话操作
      function mergeSession(info) {
        if (!info || !info.id) return;
        const i = state.sessions.findIndex((s) => s.id === info.id);
        if (i >= 0) state.sessions[i] = info; else state.sessions.unshift(info);
        state.signature = '';                  // 列表有变 → 下次 renderTabs 重绘
      }

      async function selectSession(id) {
        state.activeId = id;
        state.after = 0;
        state.signature = '';
        const log = c.querySelector('#ch-log');
        if (log) log.innerHTML = '';
        renderTabs();
        renderStatus();
        const h = await api('GET', `/api/chat/history?session_id=${encodeURIComponent(id)}`);
        mergeSession(h.session);
        appendMessages(h.messages || []);
        state.after = (h.session && h.session.last_seq) || 0;
        renderStatus();
        renderTabs();
        startTimer();
      }

      async function createSession(persist) {
        try {
          const r = await api('POST', '/api/chat/sessions', { persist: !!persist });
          mergeSession(r.session);
          await selectSession(r.session.id);
        } catch (e) { setMsg(root, e.message, true); }
      }

      async function closeSession() {
        if (!state.activeId) return;
        if (!confirm('关闭当前会话？（内存上下文释放，已落库流水不受影响）')) return;
        try {
          await api('DELETE', `/api/chat/sessions/${encodeURIComponent(state.activeId)}`);
          state.sessions = state.sessions.filter((s) => s.id !== state.activeId);
          state.activeId = '';
          state.after = 0;
          state.signature = '';
          stopTimer();
          if (state.sessions.length) await selectSession(state.sessions[0].id);
          else await createSession(state.meta ? state.meta.persist : true);
        } catch (e) { setMsg(root, e.message, true); }
      }

      // ---------------------------------------------------------------- 发送
      async function onSend() {
        const ta = c.querySelector('#ch-text');
        const text = (ta.value || '').trim();
        if (!text || !state.activeId) return;
        ta.value = '';
        try {
          const r = await api('POST', '/api/chat/send',
            { session_id: state.activeId, text });
          mergeSession(r.session);
          if (r.message) {                      // 立即回显自己这条（不等轮询）
            appendMessages([r.message]);
            state.after = Math.max(state.after, r.message.seq);
          }
          renderStatus();
          renderTabs();
          startTimer();
        } catch (e) {
          ta.value = text;                      // 发送失败：还原输入，不清空
          setMsg(root, e.message, true);
        }
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
        // 页面已被切走（DOM 被替换）→ 停表，避免后台空转。
        if (!state.activeId || !c.querySelector('#ch-log')) { stopTimer(); return; }
        try {
          const r = await api('GET',
            `/api/chat/poll?session_id=${encodeURIComponent(state.activeId)}&after=${state.after}`);
          mergeSession(r.session);
          if (r.messages && r.messages.length) appendMessages(r.messages);
          state.after = r.last_seq;
          renderStatus();
          renderTabs();
        } catch (e) {
          // 会话可能已被删除 / 核心未就绪：停止轮询并提示，别每秒刷错误。
          stopTimer();
          setMsg(root, e.message, true);
        }
      }
    },
  });
})();
