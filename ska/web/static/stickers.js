/* ===========================================================================
 * 表情包管理页（阶段五）
 *
 * 解决两个实际痛点：
 *   1) 「表情包失配」——视觉模型生成的描述/tags 不一定准，必须能人工修正；
 *   2) 「攒得慢」——从文件夹批量导入，由模型一次性打标入库。
 *
 * 关键约定：
 *   * 图片走 /api/stickers/{id}/image 原图预览（服务端只读表情包目录）；
 *   * 导入把文件读成 dataURL 分批 POST（每批 5 张），避免一次请求过大；
 *   * 服务端按 sha256 去重，重复图片只会 skipped，不会产生两份。
 *
 * 页对象通过 window.EXTRA_PAGES 注册，由 app.js 末尾并入导航。
 * =========================================================================== */

'use strict';

(function () {
  const CHUNK = 5;          // 每批提交几张（每张都会跑一次视觉，太大容易超时）
  const MAX_FILES = 200;    // 单次选择上限，防止误选整个磁盘
  const IMAGE_RE = /\.(png|jpe?g|gif|webp|bmp)$/i;

  // 把 File 读成 dataURL（服务端 _decode_upload 会剥掉 data: 头）
  function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(new Error('读取失败：' + file.name));
      reader.readAsDataURL(file);
    });
  }

  window.EXTRA_PAGES = window.EXTRA_PAGES || [];

  window.EXTRA_PAGES.push({
    id: 'stickers',
    label: '表情包管理',
    desc: '查看/编辑嵌入信息与 tags，批量导入文件夹让模型一次性打标',
    async load(root, c) {
      let items = [];

      // ---------------------------------------------------------------- 数据
      const reload = async () => {
        const d = await api('GET', '/api/stickers?limit=300');
        items = d.items || [];
        render();
      };

      // ---------------------------------------------------------------- 渲染
      function card(m) {
        return `
          <div class="sk-card" data-id="${m.id}" data-text="${esc(
            `${m.description || ''} ${m.tags || ''} ${m.emotion || ''}`.toLowerCase())}">
            <a href="/api/stickers/${m.id}/image" target="_blank" rel="noopener">
              <img src="/api/stickers/${m.id}/image" alt="" loading="lazy">
            </a>
            <div class="sk-fields">
              <label>描述</label>
              <input data-f="description" value="${esc(m.description)}">
              <label>tags</label>
              <input data-f="tags" value="${esc(m.tags)}" placeholder="空格分隔；a+b 表示同时命中">
              <label>情绪</label>
              <input data-f="emotion" value="${esc(m.emotion)}">
              <div class="inline">
                <span class="hint">使用 ${m.use_count || 0} 次</span>
                <button class="mini" data-save="${m.id}">保存</button>
                <button class="mini danger" data-del="${m.id}">删除</button>
              </div>
            </div>
          </div>`;
      }

      function render() {
        c.innerHTML = `
          <div class="panel">
            <div class="row">
              <div><label>按关键词过滤（描述 / tags / 情绪）</label>
                <input id="sk-filter" placeholder="例如：无语、猫、开心+猫"></div>
              <div class="shrink"><label>&nbsp;</label>
                <button id="sk-refresh" class="ghost">刷新</button></div>
            </div>
            <div class="spacer"></div>
            <div class="row">
              <div class="inline" style="flex-wrap:wrap">
                <input id="sk-folder" type="file" accept="image/*" multiple webkitdirectory style="display:none">
                <input id="sk-files" type="file" accept="image/*" multiple style="display:none">
                <button id="sk-pick-folder">导入文件夹</button>
                <button id="sk-pick-files" class="ghost">选择图片</button>
                <span class="hint" id="sk-progress">单次最多 ${MAX_FILES} 张</span>
              </div>
            </div>
            <p class="hint">导入时逐张调用视觉模型（GIF 先抽首帧），结果可在这里手动修正；
              同一张图按内容哈希自动去重。tags 是检索主依据，可用「a+b」表达必须同时命中。</p>
            <div class="spacer"></div>
            <p class="hint" id="sk-count"></p>
            <div class="sk-grid" id="sk-grid">${items.map(card).join('')
              || '<p class="hint">（还没有表情包，先导入一批吧）</p>'}</div>
          </div>`;

        c.querySelector('#sk-count').textContent = `共 ${items.length} 个表情包`;
        bind();
      }

      // ---------------------------------------------------------------- 交互
      function bind() {
        const grid = c.querySelector('#sk-grid');
        const filter = c.querySelector('#sk-filter');
        filter.oninput = () => {
          const kw = filter.value.trim().toLowerCase();
          grid.querySelectorAll('.sk-card').forEach((el) => {
            el.style.display = (!kw || (el.dataset.text || '').includes(kw)) ? '' : 'none';
          });
        };
        c.querySelector('#sk-refresh').onclick = () => reload().catch((e) => setMsg(root, e.message, true));

        // 只改「嵌入信息」，不动图片本身（图片内容改不了，改了就换一个表情）
        grid.querySelectorAll('[data-save]').forEach((btn) => {
          btn.onclick = async () => {
            const box = btn.closest('.sk-card');
            const id = btn.dataset.save;
            const get = (name) => box.querySelector(`[data-f="${name}"]`).value;
            try {
              await api('PUT', `/api/stickers/${id}`, {
                description: get('description'), tags: get('tags'), emotion: get('emotion'),
              });
              setMsg(root, `已保存 #${id}`);
            } catch (e) { setMsg(root, e.message, true); }
          };
        });

        grid.querySelectorAll('[data-del]').forEach((btn) => {
          btn.onclick = async () => {
            if (!confirm('删除这个表情包（连同本地文件）？')) return;
            try {
              await api('DELETE', `/api/stickers/${btn.dataset.del}`);
              setMsg(root, '已删除');
              await reload();
            } catch (e) { setMsg(root, e.message, true); }
          };
        });

        const folder = c.querySelector('#sk-folder');
        const files = c.querySelector('#sk-files');
        c.querySelector('#sk-pick-folder').onclick = () => folder.click();
        c.querySelector('#sk-pick-files').onclick = () => files.click();
        folder.onchange = () => importFiles(folder.files);
        files.onchange = () => importFiles(files.files);
      }

      // ---------------------------------------------------------------- 批量导入
      async function importFiles(fileList) {
        const progress = () => c.querySelector('#sk-progress');
        const picked = Array.from(fileList || []).filter(
          (f) => (f.type && f.type.startsWith('image/')) || IMAGE_RE.test(f.name));
        if (!picked.length) { setMsg(root, '没有可导入的图片', true); return; }
        const filesToSend = picked.slice(0, MAX_FILES);

        let added = 0, skipped = 0, failed = 0;
        for (let i = 0; i < filesToSend.length; i += CHUNK) {
          const batch = filesToSend.slice(i, i + CHUNK);
          const payload = [];
          for (const file of batch) {
            try {
              payload.push({ name: file.name, data: await readFile(file) });
            } catch (e) {
              failed += 1;
            }
          }
          if (payload.length) {
            try {
              const res = await api('POST', '/api/stickers/import', { items: payload });
              added += res.added || 0;
              skipped += res.skipped || 0;
              failed += res.failed || 0;
            } catch (e) {
              failed += payload.length;      // 整批失败按失败计，但继续下一批
            }
          }
          progress().textContent =
            `进度 ${Math.min(i + CHUNK, filesToSend.length)}/${filesToSend.length}：` +
            `新增 ${added}，重复 ${skipped}，失败 ${failed}`;
        }
        setMsg(root, `导入完成：新增 ${added}，重复跳过 ${skipped}，失败 ${failed}`);
        try { await reload(); } catch (e) { /* 列表刷新失败不影响导入结果 */ }
      }

      await reload();
    },
  });
})();
