/* SKAxx2 管理后台前端（原生 JS，无构建、无第三方库）
 *
 * 结构：
 *   1) 工具函数：api() / esc() / h()
 *   2) 引导流程：boot() → 初始化向导 / 登录 / 主界面
 *   3) 各功能页：PAGES 数组，每页一个 load(root) 函数
 *
 * 会话靠 HttpOnly Cookie 维持，前端只通过 /api/session 查询登录态。
 */

'use strict';

// ===========================================================================
// 1) 工具函数
// ===========================================================================
async function api(method, path, body) {
  const opt = { method, headers: {}, credentials: 'same-origin' };
  if (body !== undefined) {
    opt.headers['Content-Type'] = 'application/json';
    opt.body = JSON.stringify(body);
  }
  const resp = await fetch(path, opt);
  let data = {};
  try { data = await resp.json(); } catch (e) { data = {}; }
  if (!resp.ok || data.ok === false) {
    throw new Error(data.error || ('请求失败：HTTP ' + resp.status));
  }
  return data;
}

function esc(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function h(html) {
  const tpl = document.createElement('template');
  tpl.innerHTML = html.trim();
  return tpl.content.firstElementChild;
}

function setMsg(root, text, isErr) {
  const box = root.querySelector('.msg');
  if (box) { box.textContent = text; box.className = 'msg ' + (isErr ? 'err' : 'ok'); }
}

function fmtTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function table(headers, rows) {
  if (!rows.length) return '<p class="hint">（暂无数据）</p>';
  const head = headers.map((x) => `<th>${esc(x)}</th>`).join('');
  const body = rows.map((r) => '<tr>' + r.map((c) => `<td>${c}</td>`).join('') + '</tr>').join('');
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

const LEVELS = ['blacklist', 'guest', 'member', 'admin', 'owner'];

// ===========================================================================
// 2) 引导流程
// ===========================================================================
const app = document.getElementById('app');

async function boot() {
  let session = { needs_setup: false, authenticated: false };
  try { session = await api('GET', '/api/session'); } catch (e) { /* 服务未就绪 */ }
  if (session.needs_setup) return renderSetup();
  if (!session.authenticated) return renderLogin();
  renderMain(session);
}

function renderSetup() {
  app.innerHTML = `
    <div class="auth-wrap"><div class="card">
      <h1>首次使用初始化</h1>
      <p class="sub">设置管理员口令，并可选地配置 LLM 提供商与 QQ 适配器</p>
      <label>管理员账号</label>
      <input id="s-user" value="admin">
      <label>管理员口令（至少 6 位）</label>
      <input id="s-pass" type="password">
      <div class="spacer"></div>
      <div class="panel" style="padding:14px 16px">
        <h3 style="margin-bottom:6px">LLM 提供商（可跳过，稍后配置）</h3>
        <label>接口地址 base_url</label>
        <input id="s-base" placeholder="https://api.deepseek.com/v1">
        <div class="row">
          <div><label>模型</label><input id="s-model" placeholder="deepseek-chat"></div>
          <div><label>API Key</label><input id="s-key" type="password"></div>
        </div>
      </div>
      <div class="panel" style="padding:14px 16px">
        <h3 style="margin-bottom:6px">SnowLuma QQ 适配器（可跳过）</h3>
        <div class="row">
          <div><label>HTTP API 地址</label><input id="s-api" placeholder="http://127.0.0.1:3000"></div>
          <div><label>accessToken</label><input id="s-token" type="password"></div>
        </div>
        <p class="hint">填入后会自动启用 qq 渠道；事件上报地址请配到 SnowLuma 的 httpClients：
          <code>http://127.0.0.1:5098/onebot/events</code></p>
      </div>
      <button id="s-submit" style="width:100%">初始化并登录</button>
      <div class="msg"></div>
    </div></div>`;

  const root = app;
  root.querySelector('#s-submit').onclick = async () => {
    const provider = { base_url: root.querySelector('#s-base').value.trim(),
                       model: root.querySelector('#s-model').value.trim(),
                       api_key: root.querySelector('#s-key').value.trim(),
                       label: '初始化配置', id: 'default' };
    const qq = { api_base: root.querySelector('#s-api').value.trim(),
                 access_token: root.querySelector('#s-token').value.trim() };
    if (qq.api_base) { qq.enabled = true; qq.transport = 'http'; }
    try {
      await api('POST', '/api/setup', {
        username: root.querySelector('#s-user').value.trim(),
        password: root.querySelector('#s-pass').value,
        provider: provider.base_url ? provider : undefined,
        qq: qq.api_base ? qq : undefined,
      });
      boot();
    } catch (e) { setMsg(root, e.message, true); }
  };
}

function renderLogin() {
  app.innerHTML = `
    <div class="auth-wrap"><div class="card">
      <h1>登录管理后台</h1>
      <p class="sub">SKAxx2 · 本地管理界面</p>
      <label>账号</label><input id="l-user">
      <label>口令</label><input id="l-pass" type="password">
      <div class="spacer"></div>
      <button id="l-submit" style="width:100%">登录</button>
      <div class="msg"></div>
    </div></div>`;
  const root = app;
  const submit = async () => {
    try {
      await api('POST', '/api/login', {
        username: root.querySelector('#l-user').value.trim(),
        password: root.querySelector('#l-pass').value,
      });
      boot();
    } catch (e) { setMsg(root, e.message, true); }
  };
  root.querySelector('#l-submit').onclick = submit;
  root.querySelector('#l-pass').onkeydown = (ev) => { if (ev.key === 'Enter') submit(); };
}

// ===========================================================================
// 3) 主界面
// ===========================================================================
let CURRENT_SESSION = null;

function renderMain(session) {
  CURRENT_SESSION = session;
  const pages = PAGES.filter((p) => !p.ownerOnly || session.role === 'owner');
  app.innerHTML = `
    <div class="layout">
      <div class="sidebar">
        <div class="brand">SKAxx2<small>管理后台</small></div>
        <div id="nav"></div>
        <div class="nav-foot">
          <div>${esc(session.username)} · ${esc(session.role)}</div>
          <a id="logout">退出登录</a>
        </div>
      </div>
      <div class="main" id="page"></div>
    </div>`;

  const nav = document.getElementById('nav');
  pages.forEach((p) => {
    const item = h(`<div class="nav-item" data-id="${p.id}">${esc(p.label)}</div>`);
    item.onclick = () => show(p.id);
    nav.appendChild(item);
  });
  document.getElementById('logout').onclick = async () => {
    await api('POST', '/api/logout'); boot();
  };

  function show(id) {
    const page = pages.find((p) => p.id === id) || pages[0];
    // 切换页面前先执行上一页注册的清理钩子（如停止轮询定时器），
    // 避免「在线聊天 / MindLine」在后台继续 1 秒轮询、白白占用资源。
    if (typeof window.__teardown === 'function') {
      try { window.__teardown(); } catch (e) { /* 清理失败不影响切页 */ }
    }
    window.__teardown = null;
    nav.querySelectorAll('.nav-item').forEach((n) =>
      n.classList.toggle('active', n.dataset.id === page.id));
    const root = document.getElementById('page');
    root.innerHTML = `<h2>${esc(page.label)}</h2><p class="desc">${esc(page.desc)}</p>
                      <div id="content">加载中…</div><div class="msg"></div>`;
    page.load(root, root.querySelector('#content'));
  }
  window.__show = show;
  show(pages[0].id);
}

// ===========================================================================
// 4) 功能页定义
// ===========================================================================
const PAGES = [
  // ---------------------------------------------------------------- 运行状态
  {
    id: 'status', label: '运行状态', desc: '心跳、队列、渠道在线情况与数据规模',
    async load(root, c) {
      const d = await api('GET', '/api/status');
      const chans = Object.entries(d.channels).map(([name, v]) =>
        `<span class="badge ${v.online ? 'ok' : 'off'}">${esc(name)} ${v.online ? '在线' : '离线'}</span>`).join(' ');
      c.innerHTML = `
        <div class="panel"><h3>核心</h3>
          <div>当前提供商：<b>${esc(d.provider.id || '(未设置)')}</b> · 模型 ${esc(d.provider.model)}</div>
          <div>心跳计数：${d.beat.count} · bpm：${d.beat.bpm} · 待处理事件：${d.beat.queue}</div>
          <div class="spacer"></div><div>渠道：${chans || '（无）'}</div>
        </div>
        <div class="panel"><h3>数据规模</h3>
          <div>记忆库：${Object.entries(d.memory).map(([k, v]) => `${esc(k)}=${v}`).join(' · ') || '（未启用）'}</div>
          <div>管理库：${Object.entries(d.admin).map(([k, v]) => `${esc(k)}=${v}`).join(' · ')}</div>
        </div>`;
    },
  },

  // ---------------------------------------------------------------- 绑定 QQ
  {
    id: 'qq', label: '绑定 QQ 适配器', desc: '对接 SnowLuma（OneBot v11）：HTTP API + 事件上报',
    async load(root, c) {
      const cfg = await api('GET', '/api/qq/config');
      c.innerHTML = `
        <div class="panel"><h3>连接参数</h3>
          <div class="grid2">
            <div><label>HTTP API 地址（SnowLuma httpServers）</label>
              <input id="q-base" value="${esc(cfg.api_base)}" placeholder="http://127.0.0.1:3000"></div>
            <div><label>accessToken</label>
              <input id="q-token" type="password" placeholder="${cfg.has_token ? '（已设置，留空则不修改）' : '未设置'}"></div>
            <div><label>事件上报路径（配到 SnowLuma httpClients）</label>
              <input id="q-path" value="${esc(cfg.event_path)}"></div>
            <div><label>上报令牌 report_token（可留空）</label>
              <input id="q-report" value="${esc(cfg.report_token)}"></div>
            <div><label>默认群号（主动发言兜底目标）</label>
              <input id="q-group" value="${esc(cfg.default_group)}"></div>
            <div><label>默认私聊对象 user_id</label>
              <input id="q-user" value="${esc(cfg.default_user)}"></div>
          </div>
          <div class="checkbox"><input type="checkbox" id="q-enabled" ${cfg.enabled ? 'checked' : ''}>
            <label for="q-enabled" style="margin:0">启用 QQ 渠道</label></div>
          <p class="hint">当前状态：<span class="badge ${cfg.connected ? 'ok' : 'off'}">${cfg.connected ? '已连接' : '未连接'}</span>
            · 请把 <code>http://127.0.0.1:${(await api('GET', '/api/settings')).web.port}/onebot/events</code>
            填入 SnowLuma 的 <code>httpClients</code></p>
          <div class="spacer"></div>
          <div class="inline">
            <button id="q-save">保存并重连</button>
            <button class="ghost" id="q-test">测试连接</button>
            <button class="ghost" id="q-info">读取机器人账号</button>
          </div>
          <pre class="out" id="q-out" style="display:none"></pre>
        </div>`;

      const gather = () => {
        const out = {
          api_base: c.querySelector('#q-base').value.trim(),
          event_path: c.querySelector('#q-path').value.trim(),
          report_token: c.querySelector('#q-report').value.trim(),
          default_group: c.querySelector('#q-group').value.trim(),
          default_user: c.querySelector('#q-user').value.trim(),
          enabled: c.querySelector('#q-enabled').checked,
          transport: 'http',
        };
        const token = c.querySelector('#q-token').value;
        if (token) out.access_token = token;      // 留空表示不改动已有令牌
        return out;
      };
      const show = (obj) => {
        const box = c.querySelector('#q-out');
        box.style.display = 'block';
        box.textContent = JSON.stringify(obj, null, 2);
      };
      c.querySelector('#q-save').onclick = async () => {
        try { await api('PUT', '/api/qq/config', gather()); setMsg(root, '已保存并尝试重连'); await this.load(root, c); }
        catch (e) { setMsg(root, e.message, true); }
      };
      c.querySelector('#q-test').onclick = async () => {
        try { show(await api('POST', '/api/qq/test', gather())); } catch (e) { setMsg(root, e.message, true); }
      };
      c.querySelector('#q-info').onclick = async () => {
        try { show(await api('GET', '/api/qq/login-info')); } catch (e) { setMsg(root, e.message, true); }
      };
    },
  },

  // ---------------------------------------------------------------- API 管理
  {
    id: 'providers', label: 'API 管理', desc: 'LLM 提供商增删改、切换与试调用',
    async load(root, c) {
      const d = await api('GET', '/api/providers');
      const rows = d.providers.map((p) => [
        esc(p.label) + (p.is_active ? ' <span class="badge ok">使用中</span>' : ''),
        esc(p.id), esc(p.base_url), esc(p.model),
        esc(p.context_window) + ' token',        // 默认上下文窗口（参与压缩触发）
        p.has_key ? '<span class="badge">已填密钥</span>' : '<span class="badge warn">无密钥</span>',
        `<div class="inline">
           <button class="mini ghost" data-act="test" data-id="${esc(p.id)}">测试</button>
           <button class="mini ghost" data-act="use" data-id="${esc(p.id)}">启用</button>
           <button class="mini danger" data-act="del" data-id="${esc(p.id)}">删除</button>
         </div>`,
      ]);
      const opts = d.presets.map((p) => `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join('');
      c.innerHTML = `
        <div class="panel"><h3>已有提供商</h3>
          ${table(['名称', 'ID', '地址', '模型', '上下文窗口', '密钥', '操作'], rows)}
        </div>
        <div class="panel"><h3>新增提供商</h3>
          <div class="row">
            <div><label>从预设开始</label><select id="p-preset">${opts}</select></div>
            <div><label>ID</label><input id="p-id" placeholder="唯一标识"></div>
          </div>
          <div class="grid2">
            <div><label>名称</label><input id="p-label"></div>
            <div><label>模型</label><input id="p-model"></div>
            <div><label>接口地址 base_url</label><input id="p-base"></div>
            <div><label>API Key</label>
              <input id="p-key" type="password" placeholder="（已填则留空不修改）"></div>
            <div><label>默认上下文窗口（token）</label>
              <input id="p-window" type="number" min="256" step="256"
                     placeholder="例如 32768（参与压缩触发）"></div>
          </div>
          <div class="spacer"></div><button id="p-add">添加</button>
          <pre class="out" id="p-out" style="display:none"></pre>
        </div>`;

      const presetOf = (id) => d.presets.find((x) => x.id === id) || {};
      const fill = () => {
        const p = presetOf(c.querySelector('#p-preset').value);
        c.querySelector('#p-label').value = p.label || '';
        c.querySelector('#p-model').value = p.model || '';
        c.querySelector('#p-base').value = p.base_url || '';
        c.querySelector('#p-key').value = p.api_key || '';
        c.querySelector('#p-id').value = p.id || '';
        // 预设自带各模型的常见窗口值；用户可在此手改（决定压缩触发点）
        c.querySelector('#p-window').value = p.context_window || 8192;
      };
      c.querySelector('#p-preset').onchange = fill;
      fill();

      c.querySelector('#p-add').onclick = async () => {
        try {
          await api('POST', '/api/providers', {
            id: c.querySelector('#p-id').value.trim(),
            label: c.querySelector('#p-label').value.trim(),
            base_url: c.querySelector('#p-base').value.trim(),
            model: c.querySelector('#p-model').value.trim(),
            api_key: c.querySelector('#p-key').value.trim(),
            context_window: Number(c.querySelector('#p-window').value) || 8192,
          });
          setMsg(root, '已添加'); this.load(root, c);
        } catch (e) { setMsg(root, e.message, true); }
      };
      c.querySelectorAll('button[data-act]').forEach((btn) => {
        const id = btn.dataset.id, act = btn.dataset.act;
        btn.onclick = async () => {
          try {
            if (act === 'use') { await api('POST', `/api/providers/${id}/activate`); await this.load(root, c); }
            if (act === 'del') { if (!confirm('确认删除 ' + id + '？')) return; await api('DELETE', `/api/providers/${id}`); await this.load(root, c); }
            if (act === 'test') {
              const box = c.querySelector('#p-out'); box.style.display = 'block';
              box.textContent = '测试中…';
              box.textContent = JSON.stringify(await api('POST', `/api/providers/${id}/test`), null, 2);
            }
          } catch (e) { setMsg(root, e.message, true); }
        };
      });
    },
  },

  // ---------------------------------------------------------------- 用量管理
  {
    id: 'usage', label: '用量管理', desc: 'token 用量按提供商 / 模型 / 任务 / 日期聚合',
    async load(root, c) {
      let range = '7d', group = 'provider';
      const draw = async () => {
        const d = await api('GET', `/api/usage?range=${range}&group=${group}`);
        const rows = d.rows.map((r) => [esc(r.group), r.calls, r.prompt_tokens,
          r.completion_tokens, r.total_tokens]);
        const recent = d.recent.map((r) => [fmtTime(r.ts), esc(r.provider), esc(r.model),
          esc(r.task), r.total_tokens]);
        c.innerHTML = `
          <div class="panel"><h3>汇总（近 ${esc(range)}，按 ${esc(group)}）</h3>
            <div class="row">
              <div class="shrink"><label>时间范围</label>
                <select id="u-range">
                  ${['1d', '7d', '30d', '365d'].map((x) => `<option ${x === range ? 'selected' : ''}>${x}</option>`).join('')}
                </select></div>
              <div class="shrink"><label>分组</label>
                <select id="u-group">
                  ${['provider', 'model', 'task', 'day'].map((x) => `<option ${x === group ? 'selected' : ''}>${x}</option>`).join('')}
                </select></div>
            </div>
            <div class="spacer"></div>
            <div>总调用 ${d.totals.calls} 次 · 总 token ${d.totals.total_tokens}</div>
            ${table(['分组', '调用数', '输入 token', '输出 token', '合计'], rows)}
          </div>
          <div class="panel"><h3>最近调用明细</h3>
            ${table(['时间', '提供商', '模型', '任务', 'token'], recent)}
          </div>`;
        c.querySelector('#u-range').onchange = (e) => { range = e.target.value; draw(); };
        c.querySelector('#u-group').onchange = (e) => { group = e.target.value; draw(); };
      };
      await draw();
    },
  },

  // ---------------------------------------------------------------- 聊天用户
  {
    id: 'users', label: '聊天用户管理', desc: '出现过的用户、等级、拉黑与备注',
    async load(root, c) {
      const d = await api('GET', '/api/users');
      const rows = d.users.map((u) => [
        esc(u.name) || '(无昵称)', esc(u.user_id), esc(u.channel), u.msg_count,
        `<select data-level="${esc(u.channel)}|${esc(u.user_id)}" style="width:120px">
           ${LEVELS.map((l) => `<option ${l === u.level ? 'selected' : ''}>${l}</option>`).join('')}
         </select>`,
        u.blacklisted ? '<span class="badge err">已拉黑</span>' : '<span class="badge ok">正常</span>',
        fmtTime(u.last_ts),
        `<button class="mini ghost" data-save="${esc(u.channel)}|${esc(u.user_id)}">保存</button>`,
      ]);
      c.innerHTML = `<div class="panel"><h3>用户列表</h3>
        ${table(['昵称', 'user_id', '渠道', '消息数', '等级', '状态', '最近活跃', '操作'], rows)}
        <p class="hint">等级：blacklist 拉黑 / guest 访客 / member 成员 / admin 管理 / owner 主人</p>
      </div>`;
      c.querySelectorAll('button[data-save]').forEach((btn) => {
        const [channel, uid] = btn.dataset.save.split('|');
        btn.onclick = async () => {
          const level = c.querySelector(`select[data-level="${CSS.escape(btn.dataset.save)}"]`).value;
          try {
            await api('PUT', `/api/users/${encodeURIComponent(channel)}/${encodeURIComponent(uid)}`,
              { level, blacklisted: level === 'blacklist' });
            setMsg(root, '已保存');
          } catch (e) { setMsg(root, e.message, true); }
        };
      });
    },
  },

  // ---------------------------------------------------------------- 记忆管理
  {
    id: 'memory', label: '记忆管理', desc: '长期记忆、会话摘要、用户画像与原始流水',
    async load(root, c) {
      let tab = 'long';
      const draw = async () => {
        let body = '';
        if (tab === 'long') {
          const d = await api('GET', '/api/memory/long?limit=100');
          const rows = d.items.map((m) => [m.id, esc(m.content), m.importance,
            esc(m.source), esc(m.user), fmtTime(m.created_ts),
            `<button class="mini danger" data-del="${m.id}">删除</button>`]);
          body = `
            <div class="row"><div><label>新增记忆内容</label><input id="m-content"></div>
              <div class="shrink" style="max-width:120px"><label>重要性</label><input id="m-imp" value="0.5"></div>
              <div class="shrink"><label>&nbsp;</label><button id="m-add">添加</button></div></div>
            <div class="spacer"></div>
            ${table(['ID', '内容', '重要性', '来源', '用户', '时间', '操作'], rows)}`;
        } else if (tab === 'sessions') {
          const d = await api('GET', '/api/memory/sessions');
          body = table(['ID', '用户', '摘要', '开始', '结束', '操作'], d.items.map((s) => [
            s.id, esc(s.user), esc(s.summary), fmtTime(s.started_ts), fmtTime(s.ended_ts),
            `<button class="mini danger" data-del-session="${s.id}">删除</button>`]));
        } else if (tab === 'profile') {
          const d = await api('GET', '/api/memory/profile');
          body = table(['用户', '昵称', '事实', '更新时间'], d.items.map((p) => [
            esc(p.user), esc(p.nick), esc((p.facts || []).join('；')), fmtTime(p.updated_ts)]));
        } else {
          const d = await api('GET', '/api/memory/messages?limit=150');
          body = table(['ID', '角色', '用户', '渠道', '内容', '时间', '操作'], d.items.map((m) => [
            m.id, esc(m.role), esc(m.user), esc(m.channel), esc(m.content), fmtTime(m.ts),
            `<button class="mini danger" data-del-msg="${m.id}">删除</button>`]));
        }
        c.innerHTML = `
          <div class="panel">
            <div class="inline" style="margin-bottom:12px">
              ${[['long', '长期记忆'], ['sessions', '会话摘要'], ['profile', '用户画像'], ['messages', '原始流水']]
                .map(([k, label]) => `<button class="${tab === k ? '' : 'ghost'} mini" data-tab="${k}">${label}</button>`).join('')}
            </div>
            ${body}
          </div>`;
        c.querySelectorAll('button[data-tab]').forEach((b) =>
          b.onclick = () => { tab = b.dataset.tab; draw(); });
        const addBtn = c.querySelector('#m-add');
        if (addBtn) addBtn.onclick = async () => {
          try {
            await api('POST', '/api/memory/long', {
              content: c.querySelector('#m-content').value,
              importance: parseFloat(c.querySelector('#m-imp').value) || 0.5 });
            setMsg(root, '已添加'); draw();
          } catch (e) { setMsg(root, e.message, true); }
        };
        bindDelete(c, root, '[data-del]', (id) => `/api/memory/long/${id}`, draw);
        bindDelete(c, root, '[data-del-session]', (id) => `/api/memory/sessions/${id}`, draw);
        bindDelete(c, root, '[data-del-msg]', (id) => `/api/memory/messages/${id}`, draw);
      };
      await draw();
    },
  },

  // ---------------------------------------------------------------- 权限管理
  {
    id: 'roles', label: '权限管理', desc: '后台账号与聊天用户等级规则', ownerOnly: true,
    async load(root, c) {
      const [admins, roles] = await Promise.all([api('GET', '/api/admins'), api('GET', '/api/roles')]);
      const rows = admins.admins.map((a) => [
        a.id, esc(a.username),
        `<select data-role="${a.id}" style="width:120px">
           ${['owner', 'admin', 'viewer'].map((r) => `<option ${r === a.role ? 'selected' : ''}>${r}</option>`).join('')}
         </select>`,
        fmtTime(a.created_ts), fmtTime(a.last_login_ts) || '从未',
        `<div class="inline">
          <button class="mini ghost" data-save="${a.id}">保存角色</button>
          <button class="mini danger" data-del="${a.id}">删除</button></div>`,
      ]);
      c.innerHTML = `
        <div class="panel"><h3>后台账号</h3>
          ${table(['ID', '账号', '角色', '创建', '最近登录', '操作'], rows)}
          <div class="row" style="margin-top:10px">
            <div><label>新账号</label><input id="a-user"></div>
            <div><label>口令</label><input id="a-pass" type="password"></div>
            <div class="shrink"><label>角色</label><select id="a-role">
              <option>admin</option><option>viewer</option><option>owner</option></select></div>
            <div class="shrink"><label>&nbsp;</label><button id="a-add">添加</button></div>
          </div>
        </div>
        <div class="panel"><h3>聊天用户等级规则</h3>
          <div class="grid2">
            <div><label>owner 用户（QQ 号，视为最高等级）</label>
              <input id="r-owner" value="${esc(roles.owner_user)}"></div>
            <div><label>默认等级</label><select id="r-default">
              ${LEVELS.map((l) => `<option ${l === roles.default_level ? 'selected' : ''}>${l}</option>`).join('')}
            </select></div>
          </div>
          <div class="spacer"></div><button id="r-save">保存规则</button>
          <p class="hint">等级顺序（低→高）：${LEVELS.join(' < ')}</p>
        </div>`;

      c.querySelector('#a-add').onclick = async () => {
        try {
          await api('POST', '/api/admins', {
            username: c.querySelector('#a-user').value.trim(),
            password: c.querySelector('#a-pass').value,
            role: c.querySelector('#a-role').value });
          setMsg(root, '已添加'); this.load(root, c);
        } catch (e) { setMsg(root, e.message, true); }
      };
      c.querySelectorAll('button[data-save]').forEach((btn) => {
        btn.onclick = async () => {
          try {
            await api('PUT', `/api/admins/${btn.dataset.save}`,
              { role: c.querySelector(`select[data-role="${btn.dataset.save}"]`).value });
            setMsg(root, '已更新角色');
          } catch (e) { setMsg(root, e.message, true); }
        };
      });
      bindDelete(c, root, 'button[data-del]', (id) => `/api/admins/${id}`, () => this.load(root, c));
      c.querySelector('#r-save').onclick = async () => {
        try {
          await api('PUT', '/api/roles', {
            owner_user: c.querySelector('#r-owner').value.trim(),
            default_level: c.querySelector('#r-default').value });
          setMsg(root, '已保存');
        } catch (e) { setMsg(root, e.message, true); }
      };
    },
  },

  // ---------------------------------------------------------------- 插件拓展
  {
    id: 'plugins', label: '插件拓展', desc: '总线上可启停的横切能力',
    async load(root, c) {
      const d = await api('GET', '/api/plugins');
      const rows = d.plugins.map((p) => [
        esc(p.name),
        p.enabled ? '<span class="badge ok">已启用</span>' : '<span class="badge off">已停用</span>',
        p.loaded ? '<span class="badge ok">运行中</span>' : '<span class="badge off">未装载</span>',
        `<button class="mini ${p.enabled ? 'danger' : ''}" data-name="${esc(p.name)}"
           data-enable="${p.enabled ? '0' : '1'}">${p.enabled ? '停用' : '启用'}</button>`,
      ]);
      c.innerHTML = `<div class="panel"><h3>已注册插件</h3>
        ${table(['插件', '配置状态', '运行状态', '操作'], rows)}
        <p class="hint">启停会同时写入运行时设置并立即装载/卸载（scheduler 等支持安全启停）。</p>
      </div>`;
      c.querySelectorAll('button[data-name]').forEach((btn) => {
        btn.onclick = async () => {
          try {
            await api('POST', `/api/plugins/${btn.dataset.name}/toggle`,
              { enabled: btn.dataset.enable === '1' });
            setMsg(root, '已处理'); this.load(root, c);
          } catch (e) { setMsg(root, e.message, true); }
        };
      });
    },
  },

  // ---------------------------------------------------------------- 智能体权限
  {
    id: 'agent', label: '智能体权限', desc: '工具启用开关、最小可用等级与全局限制',
    async load(root, c) {
      const [tools, agent] = await Promise.all([api('GET', '/api/tools'), api('GET', '/api/agent')]);
      const rows = tools.tools.map((t) => [
        esc(t.name), `<span class="hint">${esc(t.description)}</span>`,
        `<input type="checkbox" data-en="${esc(t.name)}" ${t.enabled ? 'checked' : ''}>`,
        `<select data-lv="${esc(t.name)}" style="width:120px">
           ${LEVELS.map((l) => `<option ${l === t.min_level ? 'selected' : ''}>${l}</option>`).join('')}
         </select>`,
        t.configured ? '' : '<span class="badge">默认</span>',
        `<button class="mini ghost" data-save="${esc(t.name)}">保存</button>`,
      ]);
      c.innerHTML = `
        <div class="panel"><h3>全局开关</h3>
          <div class="checkbox"><input type="checkbox" id="g-allow" ${agent.allow_tools ? 'checked' : ''}>
            <label for="g-allow" style="margin:0">允许模型使用工具</label></div>
          <label>最大步数（AgentLoop 上限）</label>
          <input id="g-steps" type="number" min="1" max="20" value="${agent.max_steps}" style="max-width:140px">
          <div class="spacer"></div><button id="g-save">保存</button>
        </div>
        <div class="panel"><h3>工具权限</h3>
          ${table(['工具', '说明', '启用', '最小等级', '', '操作'], rows)}
        </div>`;
      c.querySelector('#g-save').onclick = async () => {
        try {
          await api('PUT', '/api/agent', {
            allow_tools: c.querySelector('#g-allow').checked,
            max_steps: parseInt(c.querySelector('#g-steps').value, 10) || 5 });
          setMsg(root, '已保存');
        } catch (e) { setMsg(root, e.message, true); }
      };
      c.querySelectorAll('button[data-save]').forEach((btn) => {
        const name = btn.dataset.save;
        btn.onclick = async () => {
          try {
            await api('PUT', `/api/tools/${encodeURIComponent(name)}`, {
              enabled: c.querySelector(`input[data-en="${CSS.escape(name)}"]`).checked,
              min_level: c.querySelector(`select[data-lv="${CSS.escape(name)}"]`).value });
            setMsg(root, '已保存');
          } catch (e) { setMsg(root, e.message, true); }
        };
      });
    },
  },

  // ---------------------------------------------------------------- 可发布目标
  {
    id: 'targets', label: '可发布目标', desc: '机器人可主动发言的群 / 用户白名单',
    async load(root, c) {
      const d = await api('GET', '/api/targets');
      const rows = d.targets.map((t) => [
        t.id, esc(t.channel),
        t.target_type === 'group' ? '群' : '用户',
        esc(t.target_id), esc(t.name),
        t.enabled ? '<span class="badge ok">启用</span>' : '<span class="badge off">停用</span>',
        esc(t.note), fmtTime(t.created_ts),
        `<div class="inline">
          <button class="mini ghost" data-toggle="${t.id}" data-en="${t.enabled ? '0' : '1'}">${t.enabled ? '停用' : '启用'}</button>
          <button class="mini danger" data-del="${t.id}">删除</button></div>`,
      ]);
      c.innerHTML = `
        <div class="panel"><h3>白名单</h3>
          ${table(['ID', '渠道', '类型', '目标 ID', '名称', '状态', '备注', '创建', '操作'], rows)}
          <p class="hint">未配置任何白名单时，主动发言退回「默认群/私聊对象」行为（阶段一逻辑）。</p>
        </div>
        <div class="panel"><h3>从 SnowLuma 拉取候选</h3>
          <div class="inline">
            <button class="ghost" id="t-fetch">拉取群与好友列表</button>
            <span class="hint">需要 QQ 已连接</span>
          </div>
          <div id="t-cands"></div>
        </div>`;
      c.querySelector('#t-fetch').onclick = async () => {
        const box = c.querySelector('#t-cands');
        box.innerHTML = '<p class="hint">拉取中…</p>';
        try {
          const r = await api('GET', '/api/qq/candidates');
          if (!r.ok) { box.innerHTML = `<p class="msg err">${esc(r.error)}</p>`; return; }
          const groups = r.groups.map((g) => [esc(g.name), esc(g.id),
            `<button class="mini" data-add="group|${esc(g.id)}|${esc(g.name)}">加入白名单</button>`]);
          const friends = r.friends.map((f) => [esc(f.name), esc(f.id),
            `<button class="mini" data-add="user|${esc(f.id)}|${esc(f.name)}">加入白名单</button>`]);
          box.innerHTML = `<h4 style="margin:14px 0 6px">群（${r.groups.length}）</h4>${table(['名称', 'ID', '操作'], groups)}
            <h4 style="margin:14px 0 6px">好友（${r.friends.length}）</h4>${table(['昵称', 'ID', '操作'], friends)}`;
          box.querySelectorAll('button[data-add]').forEach((btn) => {
            btn.onclick = async () => {
              const [type, id, name] = btn.dataset.add.split('|');
              try {
                await api('POST', '/api/targets',
                  { channel: 'qq', target_type: type, target_id: id, name, enabled: true });
                setMsg(root, '已加入白名单'); this.load(root, c);
              } catch (e) { setMsg(root, e.message, true); }
            };
          });
        } catch (e) { box.innerHTML = `<p class="msg err">${esc(e.message)}</p>`; }
      };
      bindDelete(c, root, 'button[data-del]', (id) => `/api/targets/${id}`, () => this.load(root, c));
      c.querySelectorAll('button[data-toggle]').forEach((btn) => {
        btn.onclick = async () => {
          try {
            await api('PUT', `/api/targets/${btn.dataset.toggle}`, { enabled: btn.dataset.en === '1' });
            this.load(root, c);
          } catch (e) { setMsg(root, e.message, true); }
        };
      });
    },
  },

  // ---------------------------------------------------------------- 设置
  {
    id: 'settings', label: '设置', desc: '心跳节拍、主动发言三重闸、人设与记忆参数', ownerOnly: true,
    async load(root, c) {
      const s = await api('GET', '/api/settings');
      const th = s.throttle, mem = s.memory, qq = s.qq, web = s.web, wc = s.webchat;
      c.innerHTML = `
        <div class="panel"><h3>基础</h3>
          <div class="grid2">
            <div><label>心跳 bpm（每分钟心跳次数）</label><input id="s-bpm" type="number" value="${s.app.bpm}"></div>
            <div><label>人设档</label><select id="s-persona">
              ${['local_full', 'cloud_safe'].map((p) => `<option ${p === s.app.persona_profile ? 'selected' : ''}>${p}</option>`).join('')}
            </select></div>
          </div>
          <p class="hint">工具名单：${esc(s.app.tools.join(', '))} · 插件：${esc(s.app.plugins.join(', '))} · 渠道：${esc(s.app.channels.join(', '))}</p>
        </div>
        <div class="panel"><h3>主动发言三重闸</h3>
          <div class="grid2">
            <div><label>作息起（小时）</label><input id="s-ah0" type="number" value="${th.active_hours[0]}"></div>
            <div><label>作息止（小时）</label><input id="s-ah1" type="number" value="${th.active_hours[1]}"></div>
            <div><label>每分钟主动发言上限</label><input id="s-max" type="number" value="${th.proactive_max_per_min}"></div>
            <div><label>连续沉默阈值</label><input id="s-silent" type="number" value="${th.max_silent_beats}"></div>
            <div><label>静默退避倍数</label><input id="s-backoff" type="number" value="${th.backoff_factor}"></div>
            <div><label>发送连续失败熔断阈值</label><input id="s-fail" type="number" value="${th.send_fail_limit}"></div>
          </div>
        </div>
        <div class="panel"><h3>记忆</h3>
          <div class="grid2">
            <div><label>工作记忆窗口（条）</label><input id="s-win" type="number" value="${mem.working_window}"></div>
            <div><label>召回条数 top_k</label><input id="s-topk" type="number" value="${mem.recall_top_k}"></div>
            <div><label>会话摘要触发消息数</label><input id="s-maxmsgs" type="number" value="${mem.session_max_msgs}"></div>
            <div><label>会话空闲触发（分钟）</label><input id="s-idle" type="number" value="${mem.session_idle_min}"></div>
            <div><label>时间衰减半衰期（天）</label><input id="s-half" type="number" value="${mem.recall_half_life_days}"></div>
          </div>
        </div>
        <div class="panel"><h3>在线聊天（阶段三）</h3>
          <div class="grid2">
            <div><label>聊天身份 identity（决定权限等级与记忆归属）</label>
              <input id="s-wc-id" value="${esc(wc.identity || wc.user || 'owner')}"></div>
            <div><label>上下文窗口覆盖（0 = 跟随当前提供商）</label>
              <input id="s-wc-window" type="number" value="${wc.context_window}"></div>
            <div><label>轮询间隔 poll_ms（毫秒）</label>
              <input id="s-wc-poll" type="number" value="${wc.poll_ms}"></div>
            <div><label>压缩阈值 compress_at（0~1，占比到达即压缩）</label>
              <input id="s-wc-compress" type="number" step="0.05" value="${wc.compress_at}"></div>
          </div>
          <div class="checkbox"><input type="checkbox" id="s-wc-persist" ${wc.persist ? 'checked' : ''}>
            <label for="s-wc-persist" style="margin:0">新建会话默认保存记忆（关 = 沙盒：照常推理但不落库）</label></div>
          <p class="hint">身份默认 owner（实验平台主人）；沙盒会话适合边聊边调试、教智能体，不污染长期记忆。</p>
        </div>
        <div class="panel"><h3>后台服务</h3>
          <div class="grid2">
            <div><label>监听地址</label><input id="s-host" value="${esc(web.host)}"></div>
            <div><label>端口（修改需重启进程）</label><input id="s-port" type="number" value="${web.port}"></div>
            <div><label>会话有效期（小时）</label><input id="s-hours" type="number" value="${web.session_hours}"></div>
          </div>
        </div>
        <button id="s-save">保存设置</button>`;

      const num = (id) => parseInt(c.querySelector(id).value, 10) || 0;
      c.querySelector('#s-save').onclick = async () => {
        try {
          await api('PUT', '/api/settings', {
            app: { bpm: num('#s-bpm'), persona_profile: c.querySelector('#s-persona').value },
            throttle: {
              active_hours: [num('#s-ah0'), num('#s-ah1')],
              proactive_max_per_min: num('#s-max'), max_silent_beats: num('#s-silent'),
              backoff_factor: num('#s-backoff'), send_fail_limit: num('#s-fail') },
            memory: { working_window: num('#s-win'), recall_top_k: num('#s-topk'),
              session_max_msgs: num('#s-maxmsgs'), session_idle_min: num('#s-idle'),
              recall_half_life_days: num('#s-half') },
            web: { host: c.querySelector('#s-host').value.trim(), port: num('#s-port'),
              session_hours: num('#s-hours') },
            qq: { api_base: qq.api_base },
            webchat: {
              identity: c.querySelector('#s-wc-id').value.trim() || 'owner',
              persist: c.querySelector('#s-wc-persist').checked,
              poll_ms: num('#s-wc-poll') || 1000,
              compress_at: parseFloat(c.querySelector('#s-wc-compress').value) || 0.8,
              context_window: num('#s-wc-window'),
            },
          });
          setMsg(root, '已保存（部分项下次心跳/重启生效）');
        } catch (e) { setMsg(root, e.message, true); }
      };
    },
  },
];

// 并入外部页面脚本注册的页（chat.js / mindline.js 先于本文件加载，
// 各自把页对象 push 进 window.EXTRA_PAGES，这里统一合入导航）。
if (Array.isArray(window.EXTRA_PAGES)) {
  PAGES.push(...window.EXTRA_PAGES);
}

// 通用的「删除按钮」绑定
function bindDelete(container, root, selector, urlOf, after) {
  container.querySelectorAll(selector).forEach((btn) => {
    btn.onclick = async () => {
      if (!confirm('确认删除？')) return;
      try { await api('DELETE', urlOf(btn.dataset.del)); setMsg(root, '已删除'); after(); }
      catch (e) { setMsg(root, e.message, true); }
    };
  });
}

boot();