<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { resolve, asset } from '$app/paths';
  import { locale } from '$lib/i18n';
  import { outcome, type Gallery, type Sample } from '$lib/vision/types';
  import '$lib/vision/visual.css';
  const say = (ko: string, en: string, ja: string) => $locale === 'ko' ? ko : $locale === 'ja' ? ja : en;
  const material = (id: string | null) => id === null ? say('보류', 'Abstained', '保留') : ({ cardboard: say('골판지','Cardboard','段ボール'), glass: say('유리','Glass','ガラス'), metal: say('금속','Metal','金属'), paper: say('종이','Paper','紙'), plastic: say('플라스틱','Plastic','プラスチック'), trash: say('일반 쓰레기','Trash','その他ごみ') }[id] ?? id);
  const statusLabel = (status: string) => status === 'correct' ? say('정답','Correct','正解') : status === 'wrong' ? say('오답','Wrong','不正解') : say('보류','Abstained','保留');
  let data = $state<Gallery>();
  let failed = $state(false);
  let modelId = $state('qwen3vl');
  let category = $state('all');
  let status = $state('all');
  let selected = $state<Sample>();
  let pending = $state(true);
  const model = $derived(data?.models.find((item) => item.id === modelId));
  const rows = $derived(Object.values(model?.observations ?? {}));
  const accepted = $derived(rows.filter((row) => row.selected !== null));
  const correct = $derived(rows.filter((row) => outcome(row) === 'correct').length);
  const rawCorrect = $derived(rows.filter((row) => row.raw_top1 === row.ground_truth).length);
  const filtered = $derived(data?.source.records.filter((sample) => (category === 'all' || category === sample.label) && (status === 'all' || (model && outcome(model.observations[sample.name]) === status))) ?? []);
  const detail = $derived(selected && model?.observations[selected.name]);
  async function choose(sample: Sample) { selected = sample; await tick(); document.getElementById('photo-detail')?.scrollIntoView({ block: 'start' }); }
  function updateFilters() { selected = undefined; }
  async function load() {
    pending = true; failed = false;
    try {
      const result = await fetch(asset('/trashnet/recorded.json'));
      if (!result.ok) throw new Error('recording');
      data = await result.json() as Gallery;
      selected = data.source.records[0];
    } catch { failed = true; } finally { pending = false; }
  }
  onMount(() => { void load(); });
</script>
<svelte:head><title>TrashNet · {say('사진으로 보는 분류','Visual classification','写真で見る分類')} · L2S1</title><meta name="description" content="Explore 120 TrashNet photos with real L2S1 classification results, abstentions, and per-class comparisons across three models." /></svelte:head>
<main class="visual-page">
  <p class="eyebrow">VISION LAB / 01 — TRASHNET</p>
  <h1>{say('사진을 보고, 판단을 비교하세요.','See the image. Compare the decision.','写真を見て、判断を比べる。')}</h1>
  <p class="lead">{say('유리병부터 구겨진 종이까지. 120장의 실제 사진에서 모델이 무엇을 맞히고, 틀리고, 보류했는지 직접 살펴보세요.','From glass bottles to crumpled paper. Explore what each model gets right, gets wrong, or abstains on across 120 real photographs.','ガラス瓶から丸めた紙まで。120枚の実際の写真で、各モデルの正解・誤り・保留を確認できます。')}</p>
  <nav class="tabs" aria-label={say('시각 예제','Visual examples','視覚サンプル')}><a href={resolve('/trashnet')} aria-current="page">TrashNet</a><a href={resolve('/detect')}>Detect ↗</a><a href={resolve('/demo')}>{say('이미지 판단 실험','Image playground','画像判断')}</a></nav>
  {#if pending}<p role="status">{say('사진과 실행 기록을 불러오는 중…','Loading photos and recorded results…','写真と実行記録を読み込み中…')}</p>
  {:else if failed}<div class="error" role="alert">{say('기록을 불러오지 못했습니다.','Could not load the recording.','記録を読み込めませんでした。')} <button onclick={load}>{say('다시 시도','Retry','再試行')}</button></div>
  {:else if data && model}
    <div class="controls">
      <label>{say('비교할 모델','Model','モデル')}<select aria-label={say('비교할 모델','Model','モデル')} bind:value={modelId}><option value="qwen3vl">Qwen3-VL 2B</option><option value="gemma4">Gemma 4 E2B</option><option value="smolvlm">SmolVLM 256M</option></select></label>
      <label>{say('정답 재질','Ground-truth material','正解の素材')}<select aria-label={say('정답 재질','Ground-truth material','正解の素材')} bind:value={category} onchange={updateFilters}><option value="all">{say('전체 재질','All materials','すべての素材')}</option>{#each data.source.classes as id (id)}<option value={id}>{material(id)}</option>{/each}</select></label>
      <label>{say('판단 결과','Outcome','判断結果')}<select aria-label={say('판단 결과','Outcome','判断結果')} bind:value={status} onchange={updateFilters}><option value="all">{say('전체 결과','All outcomes','すべての結果')}</option>{#each ['correct','wrong','abstained'] as id (id)}<option value={id}>{statusLabel(id)}</option>{/each}</select></label>
      <span class="muted count">{filtered.length} / 120 {say('장','photos','枚')}</span>
    </div>
    <div class="stats" aria-label={say('전체 120장 집계','All 120 photos','全120枚の集計')}>
      <div class="stat"><span>{say('전체 중 수락 정답','Accepted correct / all','全体の受理正解')}</span><strong>{correct}/120</strong><span>{(correct / 120 * 100).toFixed(1)}%</span></div>
      <div class="stat"><span>{say('응답 수락률','Coverage','回答受理率')}</span><strong>{accepted.length}/120</strong><span>{(accepted.length / 120 * 100).toFixed(1)}%</span></div>
      <div class="stat"><span>{say('수락한 답 중 정답','Correct / accepted','受理した回答の正解')}</span><strong>{accepted.length ? `${(correct / accepted.length * 100).toFixed(1)}%` : '—'}</strong><span>{correct}/{accepted.length}</span></div>
      <div class="stat"><span>{say('보류 전 1순위 정답','Raw top-1 correct','保留前の首位正解')}</span><strong>{rawCorrect}/120</strong><span>{say('보류도 포함한 후보 순위','Includes abstained rankings','保留した順位も含む')}</span></div>
    </div>
    <p class="provenance"><strong>{say('실제 모델 실행 기록','Recorded model inference','実際のモデル実行記録')}</strong> · {data.recorded_at} · {data.runtime}<br />{say('집계는 필터와 관계없이 전체 120장 기준입니다. 사진 선택은 저장된 결과를 표시하며 모델을 새로 실행하지 않습니다.','Metrics always cover all 120 photos. Selecting a photo displays saved results; it does not run inference.','集計はフィルターにかかわらず全120枚が対象です。写真の選択は保存結果の表示であり、再推論ではありません。')}</p>
    {#if selected && detail}
      <section id="photo-detail" class="split detail" aria-label={say('선택한 사진 상세','Selected photo details','選択した写真の詳細')}>
        <div class="panel photo-panel"><img class="hero-photo" src={asset(selected.image_url)} alt={`${material(selected.label)} · ${selected.name.split('/').at(-1)}`} width="512" height="384" /><div class="photo-caption"><span>{selected.name.split('/').at(-1)}</span><a href={resolve(`/detect?sample=${encodeURIComponent(selected.name.split('/').at(-1) ?? '')}`)}>{say('이 사진에서 객체 탐지','Detect objects in this photo','この写真の物体を検出')} ↗</a></div></div>
        <div class="panel"><span class={`badge ${outcome(detail)}`}>{statusLabel(outcome(detail))}</span><h2 class="decision">{material(detail.selected)}</h2><p>{say('데이터셋 정답','Dataset label','データセットの正解')} <strong>{material(selected.label)}</strong></p><p class="muted">{model.name} · {detail.latency_ms.toFixed(1)} ms · {say('기록된 처리 시간','recorded latency','記録した処理時間')}</p>
          {#each [...detail.scores].sort((a,b) => b.option_probability - a.option_probability) as score (score.id)}<div class="bar-row"><span>{material(score.id)}</span><div class="track"><span style:width={`${score.option_probability * 100}%`}></span></div><strong class="number">{(score.option_probability * 100).toFixed(1)}%</strong></div>{/each}
          <p class="muted">{say('후보 내 상대 점수이며 정답일 확률은 아닙니다.','Relative scores among candidates, not probabilities of correctness.','候補間の相対スコアであり、正解確率ではありません。')}</p>
          {#if detail.selected === null}<p class="badge abstained">{say('판단 보류','Decision abstained','判断を保留')} · {detail.abstention_reasons.join(', ')}</p>{/if}
          <p class="muted">Candidate mass: {detail.candidate_mass.toPrecision(5)}<br />{say('수락 기준','Acceptance thresholds','受理基準')}: top ≥ 0.8 · mass ≥ 0.05</p>
          <div class="comparison">{#each data.models as peer (peer.id)}{@const row = peer.observations[selected.name]}<div><span>{peer.name}</span><strong class={`badge ${outcome(row)}`}>{material(row.selected)}</strong></div>{/each}</div>
          <details><summary>{say('원본 점수와 기록 보기','View original scores and record','元のスコアと記録を見る')}</summary><pre>{JSON.stringify(detail, null, 2)}</pre></details>
        </div>
      </section>
    {/if}
    <h2 class="gallery-heading">{say('사진 둘러보기','Explore the photographs','写真を探す')} <span>{filtered.length}</span></h2>
    <div class="gallery">
      {#each filtered as sample (sample.name)}{@const row = model.observations[sample.name]}
        <button class="photo-card" class:chosen={selected?.name === sample.name} aria-pressed={selected?.name === sample.name} onclick={() => choose(sample)} aria-label={`${sample.name.split('/').at(-1)} · ${material(sample.label)} · ${statusLabel(outcome(row))}`}>
          <img src={asset(sample.image_url)} alt={material(sample.label)} loading="lazy" width="512" height="384" />
          <div class="card-body"><span class={`badge ${outcome(row)}`}>{statusLabel(outcome(row))}</span><strong>{material(sample.label)} <span aria-hidden="true">→</span> {material(row.selected)}</strong><small>{sample.name.split('/').at(-1)}</small></div>
        </button>
      {:else}<p class="empty">{say('이 조건에 맞는 사진이 없습니다.','No photos match these filters.','この条件に合う写真はありません。')}</p>{/each}
    </div>
    <details class="panel matrix"><summary>{say('재질별 혼동 행렬 보기','Show confusion matrix by material','素材別の混同行列を見る')}</summary><div class="table-scroll"><table><caption>{say('행: 정답 재질 / 열: 모델의 수락 답변 또는 보류','Rows: ground truth / columns: accepted answer or abstention','行：正解素材／列：受理した回答または保留')}</caption><thead><tr><th scope="col">{say('정답 ↓ / 결과 →','Label ↓ / Result →','正解 ↓ / 結果 →')}</th>{#each [...data.source.classes, 'abstained'] as label (label)}<th scope="col">{material(label === 'abstained' ? null : label)}</th>{/each}</tr></thead><tbody>{#each data.source.classes as truth (truth)}<tr><th scope="row">{material(truth)}</th>{#each [...data.source.classes, 'abstained'] as predicted (predicted)}{@const count = rows.filter((row) => row.ground_truth === truth && (row.selected ?? 'abstained') === predicted).length}<td class:diagonal={truth === predicted} class:populated={count > 0}>{count}</td>{/each}</tr>{/each}</tbody></table></div></details>
    <footer class="footer">{say('각 재질 20장씩 고정 추출한 예제입니다. 전체 데이터셋 성능이나 실사용 정확도를 뜻하지 않습니다.','A fixed sample of 20 photos per material; this is not full-dataset or deployment accuracy.','各素材20枚の固定サンプルです。全データセットや実運用の精度ではありません。')}<br /><a href={`https://github.com/garythung/trashnet/tree/${data.source.source_commit}`} target="_blank" rel="noreferrer">TrashNet · Gary Thung / Mindy Yang</a> · <a href={asset('/trashnet/THIRD_PARTY_NOTICE.txt')} rel="external">MIT / {say('이미지 출처','Image credits','画像の出典')}</a> · <a href={asset('/trashnet/recorded.json')} download>{say('원본 기록 내려받기','Download records','記録をダウンロード')}</a></footer>
  {/if}
</main>
<style>
  .count{margin-left:auto}.detail{margin-bottom:32px}.photo-panel{padding:12px!important}.hero-photo{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;background:#f5f5f3;border-radius:8px}.photo-caption{display:flex;flex-wrap:wrap;justify-content:space-between;gap:12px;font-size:12px;padding:14px 8px 6px}.decision{font-size:34px!important;margin:15px 0!important}.comparison{border-top:1px solid var(--theme-border);padding-top:12px}.comparison>div{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:9px 0;font-size:12px}.gallery-heading{display:flex;gap:12px;align-items:center}.gallery-heading span{font-size:12px;color:var(--theme-muted)}.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}.photo-card{padding:0!important;text-align:left;overflow:hidden;border-radius:10px!important}.photo-card.chosen{outline:3px solid var(--theme-chart);outline-offset:2px}.photo-card img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:cover;background:white}.card-body{padding:13px}.card-body strong,.card-body small{display:block;margin-top:10px;font-size:12px}.card-body small{color:var(--theme-muted);font-size:10px}.matrix{margin-top:28px}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;text-align:center;margin-top:16px}caption{font-size:12px;text-align:left;padding:15px 0;color:var(--theme-muted)}th,td{padding:12px;border:1px solid var(--theme-border);white-space:nowrap}td.populated{background:var(--theme-warning-bg)}td.diagonal{background:var(--theme-tint);font-weight:bold}@media(max-width:1000px){.gallery{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:650px){.gallery{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.count{margin-left:0}.card-body{padding:10px}}
</style>
