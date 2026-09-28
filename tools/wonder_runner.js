// CDC WONDER export runner for the deaths-of-despair revisit.
//
// WONDER has no API for county-level data, so the exports in data/wonder/ were
// produced in a browser. To regenerate them: open the WONDER request form for a
// database (for example https://wonder.cdc.gov/ucd-icd10.html), accept the data
// use restrictions yourself, paste this file into the browser console, then run
//
//   const specs = /* the entries of data/wonder/queries.json for this database */;
//   await dodWonder.runAll(specs);
//
// Each query is posted with the form's own session, the tab-separated export is
// checked (it must be an attachment, and WONDER's echoed "Query Parameters"
// must list every requested year), and the file is saved as a download named
// dod__<path with slashes as double underscores>.tsv. Move those into
// data/wonder/ at the paths given by each spec's "file".
(() => {
  const RACE = { D76: 'V8', D77: 'V8', D158: 'V42', D157: 'V42' };
  const groupVar = (db) => ({
    county: `${db}.V9-level2`, state: `${db}.V9-level1`, year: `${db}.V1-level1`,
    age10: `${db}.V5`, hispanic: `${db}.V17`, race: `${db}.${RACE[db]}`,
    icd: `${db}.V2-level3`, mcd: `${db}.V13-level3`,
  });

  function requestForm() {
    const form = [...document.forms].find((f) => /datarequest/.test(f.action));
    if (!form) throw new Error('No WONDER request form on this page (accept the data use restrictions first).');
    return form;
  }

  function build(form, spec) {
    const db = form.elements.dataset_code.value;
    if (db !== spec.db) throw new Error(`${spec.file} targets ${spec.db}; this page is ${db}.`);
    const fd = new FormData(form);
    const many = (k, vals) => { fd.delete(k); vals.forEach((v) => fd.append(k, String(v))); };
    const g = groupVar(db);
    spec.groupBy.forEach((k, i) => fd.set(`B_${i + 1}`, g[k]));
    for (let i = spec.groupBy.length + 1; i <= 5; i++) fd.set(`B_${i}`, '*None*');
    many(`F_${db}.V1`, spec.years);
    fd.set('O_ucd', `${db}.V2`);
    many(`F_${db}.V2`, spec.ucd);
    if (spec.mcd) {
      // The multiple-cause finder runs in WONDER's advanced mode (O_V13_fmode
      // = fadv): selected codes go in the V_ text box, not the F_ list.
      fd.set('O_mcd', `${db}.V13`);
      fd.set(`V_${db}.V13`, spec.mcd.join('\n'));
      many(`F_${db}.V13`, ['*All*']);
    }
    if (spec.ages) { fd.set('O_age', `${db}.V5`); many(`V_${db}.V5`, spec.ages); }
    if (spec.aar) {
      fd.set('O_aar_enable', 'true'); fd.set('O_aar_CI', 'true');
      fd.set('O_aar', 'aar_std'); fd.set('O_aar_pop', '0000');
    } else {
      fd.delete('O_aar_enable'); fd.delete('O_aar_CI'); fd.set('O_aar', 'aar_none');
    }
    fd.delete('O_show_totals'); fd.delete('O_show_zeros'); fd.set('O_show_suppressed', 'true');
    fd.set('O_precision', '2'); fd.set('O_timeout', '600');
    fd.set('O_export-format', 'tsv');
    fd.set('O_change_action-Send-Export Results', 'Export Results');
    fd.set('action-Send', 'Send');
    return new URLSearchParams(fd);
  }

  function save(name, text) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/tab-separated-values' }));
    a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function run(spec) {
    const form = requestForm();
    let res;
    for (let attempt = 0; ; attempt++) {
      res = await fetch(form.action, { method: 'POST', body: build(form, spec), credentials: 'include' });
      if (res.status !== 429 || attempt === 4) break;
      await sleep(60000 * (attempt + 1)); // WONDER rate limit: back off 1, 2, 3, 4 minutes
    }
    const text = await res.text();
    if (!res.ok || !/attachment/.test(res.headers.get('content-disposition') || '')) {
      throw new Error(`${spec.file}: HTTP ${res.status}, WONDER returned a page, not an export.`);
    }
    const lines = text.split('\n');
    const notes = lines.findIndex((l) => l.startsWith('"---"'));
    const params = lines.slice(notes).filter((l) => /^"(Query Parameters|ICD-10|MCD|UCD|Multiple Cause|Year\/Month|Group By|Ten-Year|Standard Population)/.test(l));
    const data = lines.slice(1, notes).filter((l) => l.startsWith('\t'));
    // WONDER echoes the selected years, except when a request spans the
    // database's whole range; then, if grouped by year, check the rows instead.
    const yearsLine = params.find((l) => l.startsWith('"Year/Month')) || '';
    const rowsText = data.join('\n');
    const applied = (y) => yearsLine
      ? yearsLine.includes(String(y))
      : spec.groupBy.includes('year') && new RegExp(`\\t"?${y}"?\\t`).test(rowsText);
    const missing = spec.years.filter((y) => !applied(y));
    if (missing.length) throw new Error(`${spec.file}: WONDER did not apply years ${missing.join(', ')}.`);
    if (spec.mcd) {
      const echo = lines.slice(notes).join(' ');
      const mcdAt = echo.indexOf('MCD - ICD-10 Codes');
      const absent = spec.mcd.filter((c) => mcdAt < 0 || !echo.slice(mcdAt).includes(c));
      if (absent.length) throw new Error(`${spec.file}: WONDER did not apply multiple cause ${absent.join(', ')}.`);
    }
    save(`dod__${spec.file.replace(/\//g, '__')}`, text);
    return { file: spec.file, bytes: text.length, rows: data.length, suppressed: data.filter((l) => l.includes('Suppressed')).length, params };
  }

  async function runAll(specs, pauseMs = 15000) {
    const log = [];
    for (const spec of specs) {
      try { log.push(await run(spec)); } catch (e) { log.push({ file: spec.file, error: String(e.message || e) }); }
      window.dodWonderLog = log;
      await sleep(pauseMs);
    }
    return log;
  }

  window.dodWonder = { build, run, runAll };
})();
