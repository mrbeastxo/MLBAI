const gamesRoot = document.querySelector("#games");
const notice = document.querySelector("#notice");
const dateInput = document.querySelector("#game-date");
const dialog = document.querySelector("#game-dialog");
const dialogContent = document.querySelector("#dialog-content");
const resultRows = document.querySelector("#result-rows");
const resultLimit = 25;
let resultOffset = 0;
let resultTeam = "";

const percent = (value, digits = 1) => value == null ? "Pending" : `${(value * 100).toFixed(digits)}%`;
const label = (value) => String(value || "unknown").replaceAll("_", " ");
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, character => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
})[character]);

function localToday() {
  const now = new Date();
  const offset = now.getTimezoneOffset() * 60_000;
  return new Date(now - offset).toISOString().slice(0, 10);
}

function renderLoading() {
  gamesRoot.innerHTML = Array.from({ length: 6 }, () => '<div class="skeleton"></div>').join("");
  notice.classList.add("hidden");
}

function gameCard(game) {
  const homeLean = game.model_lean === game.home_team;
  const awayLean = game.model_lean === game.away_team;
  const maxProbability = Math.max(game.home_win_probability, game.away_win_probability);
  const start = new Date(game.game_time_utc).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  return `
    <button class="game-card" data-game-id="${escapeHtml(game.game_id)}">
      <span class="card-top">
        <span class="game-time">${escapeHtml(start)}</span>
        <span class="evidence">${escapeHtml(label(game.evidence_grade))}</span>
      </span>
      <span class="teams">
        <span class="team-row ${awayLean ? "lean" : ""}">
          <span class="team-name">${escapeHtml(game.away_team)}</span>
          <span class="probability">${percent(game.away_win_probability)}</span>
        </span>
        <span class="team-row ${homeLean ? "lean" : ""}">
          <span class="team-name">${escapeHtml(game.home_team)}</span>
          <span class="probability">${percent(game.home_win_probability)}</span>
        </span>
      </span>
      ${game.away_expected_runs != null && game.home_expected_runs != null
        ? `<span class="projected-score">Projected score · ${game.away_expected_runs.toFixed(1)}–${game.home_expected_runs.toFixed(1)}</span>`
        : ""}
      <span class="confidence-track"><span class="confidence-fill" style="width:${maxProbability * 100}%"></span></span>
      <span class="card-footer"><span>${escapeHtml(label(game.certainty_band))}</span><span>View analysis →</span></span>
    </button>`;
}

function factorList(items, emptyText) {
  if (!items.length) return `<p class="muted">${emptyText}</p>`;
  return `<div class="factor-list">${items.map(item => `
    <div class="factor">
      <strong>${escapeHtml(item.factor)}</strong>
      <span>Game difference: ${escapeHtml(item.raw_home_minus_away)} · impact ${item.log_odds_contribution > 0 ? "+" : ""}${item.log_odds_contribution.toFixed(3)}</span>
    </div>`).join("")}</div>`;
}

function starterCard(side, starter) {
  if (!starter?.announced) return `<div class="context-card"><span>${side} starter</span><strong>Not announced</strong><small>MLB has not confirmed a probable starter.</small></div>`;
  const stats = [starter.era ? `ERA ${starter.era}` : null, starter.whip ? `WHIP ${starter.whip}` : null, starter.days_rest != null ? `${starter.days_rest} days rest` : null].filter(Boolean).join(" · ");
  return `<div class="context-card"><span>${side} starter</span><strong>${escapeHtml(starter.name || "Unknown")}</strong><small>${escapeHtml(stats || "Season stats unavailable")}</small></div>`;
}

function bullpenCard(side, bullpen) {
  if (!bullpen?.available) return `<div class="context-card"><span>${side} bullpen</span><strong>Unavailable</strong><small>Workload snapshot was not available.</small></div>`;
  return `<div class="context-card"><span>${side} bullpen</span><strong>Workload ${escapeHtml(bullpen.workload_index)}</strong><small>${escapeHtml(bullpen.pitches_last_3_days)} pitches in 3 days · ${escapeHtml(bullpen.relievers_back_to_back)} back-to-back</small></div>`;
}

function showGame(game) {
  const leanProbability = game.model_lean === game.home_team ? game.home_win_probability : game.away_win_probability;
  dialogContent.innerHTML = `
    <article class="detail">
      <p class="eyebrow">Game analysis</p>
      <h3>${escapeHtml(game.away_team)}<br />at ${escapeHtml(game.home_team)}</h3>
      <p class="muted">${new Date(game.game_time_utc).toLocaleString([], { dateStyle: "long", timeStyle: "short" })}</p>
      <div class="detail-score">
        <span>Model lean</span>
        <strong>${escapeHtml(game.model_lean)} · ${percent(leanProbability)}</strong>
        <span>${escapeHtml(label(game.certainty_band))} · ${escapeHtml(label(game.evidence_grade))}</span>
      </div>
      ${game.away_expected_runs != null && game.home_expected_runs != null ? `<div class="score-projection">
        <span>Expected runs</span>
        <strong>${escapeHtml(game.away_team)} ${game.away_expected_runs.toFixed(1)} · ${escapeHtml(game.home_team)} ${game.home_expected_runs.toFixed(1)}</strong>
        <small>Projected total ${game.expected_total_runs.toFixed(1)} · model average, not an exact score</small>
      </div>` : ""}
      ${game.outcome_uncertainty ? `<section class="uncertainty-section">
        <div class="context-title"><h4>Outcome uncertainty</h4><span>Analysis context only</span></div>
        <div class="uncertainty-grid">
          <div><span>Most likely score</span><strong>${game.outcome_uncertainty.most_likely_score.away}–${game.outcome_uncertainty.most_likely_score.home}</strong></div>
          <div><span>Extra innings</span><strong>${percent(game.outcome_uncertainty.extra_innings_probability)}</strong></div>
          <div><span>${escapeHtml(game.away_team)} range</span><strong>${game.outcome_uncertainty.away_runs_80_percent_range.join("–")} runs</strong></div>
          <div><span>${escapeHtml(game.home_team)} range</span><strong>${game.outcome_uncertainty.home_runs_80_percent_range.join("–")} runs</strong></div>
        </div>
        <p class="muted context-note">${escapeHtml(game.outcome_uncertainty.note)}</p>
      </section>` : ""}
      <div class="factor-columns">
        <section><h4>Supporting the lean</h4>${factorList(game.strongest_supporting_factors, "No strong supporting factor.")}</section>
        <section><h4>Working against it</h4>${factorList(game.strongest_opposing_factors, "No strong opposing factor.")}</section>
      </div>
      ${game.matchup_context ? `<section class="context-section">
        <div class="context-title"><h4>Pitching context</h4><span>Context only · not in probability</span></div>
        <div class="context-grid">
          ${starterCard("Away", game.matchup_context.away_starter)}
          ${starterCard("Home", game.matchup_context.home_starter)}
          ${bullpenCard("Away", game.matchup_context.away_bullpen)}
          ${bullpenCard("Home", game.matchup_context.home_bullpen)}
        </div>
        <p class="muted context-note">${escapeHtml(game.matchup_context.note)}</p>
      </section>` : ""}
      <p class="reliability">${escapeHtml(game.reliability_note)}</p>
    </article>`;
  dialog.showModal();
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

async function loadPerformance() {
  try {
    const data = await fetchJson("/api/v1/performance");
    document.querySelector("#tracked-count").textContent = data.settled_games + data.pending_games;
    document.querySelector("#accuracy").textContent = percent(data.accuracy);
    document.querySelector("#score-mae").textContent = data.score_mae == null ? "Pending" : data.score_mae.toFixed(2);
    document.querySelector("#score-sample").textContent = data.score_projection_games
      ? `${data.score_projection_games} settled projection${data.score_projection_games === 1 ? "" : "s"}`
      : "No settled projections";
    const integrity = document.querySelector("#ledger-status");
    const valid = data.hash_chain_valid && data.score_projection_hashes_valid;
    integrity.textContent = valid ? "Verified" : "Check failed";
    integrity.classList.toggle("good", valid);
  } catch {
    document.querySelector("#ledger-status").textContent = "Unavailable";
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes;
  let unit = "B";
  for (const candidate of units) {
    value /= 1024;
    unit = candidate;
    if (value < 1024) break;
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${unit}`;
}

async function loadSystemHealth() {
  try {
    const data = await fetchJson("/api/v1/system");
    const scheduler = data.scheduler;
    document.querySelector("#schedule-time").textContent = scheduler.installed ? `${scheduler.schedule} daily` : "Not installed";
    document.querySelector("#next-run").textContent = scheduler.next_run_local
      ? `Next: ${new Date(scheduler.next_run_local).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}`
      : "No scheduled run";
    const lastRun = data.last_run;
    document.querySelector("#last-run-status").textContent = lastRun ? label(lastRun.status) : "Waiting";
    document.querySelector("#last-run-detail").textContent = lastRun
      ? `${lastRun.date} · ${lastRun.predictions_generated ?? 0} predictions`
      : "Runs after the next schedule";
    document.querySelector("#storage-used").textContent = formatBytes(data.storage.data_bytes);
    document.querySelector("#log-health").textContent = data.logs.has_errors ? "Needs review" : "Clear";
    document.querySelector("#log-detail").textContent = data.logs.has_errors
      ? `${formatBytes(data.logs.error_bytes)} in error log`
      : "No scheduler errors recorded";
    const pill = document.querySelector("#system-pill");
    const healthy = scheduler.installed && !data.logs.has_errors && (!lastRun || lastRun.status === "success");
    pill.textContent = healthy ? "Automation healthy" : "Check automation";
    pill.classList.toggle("healthy", healthy);
  } catch {
    document.querySelector("#system-pill").textContent = "Health unavailable";
  }
}

async function loadCalibration() {
  try {
    const data = await fetchJson("/api/v1/calibration");
    document.querySelector("#raw-log-loss").textContent = data.raw_model.metrics.log_loss.toFixed(4);
    document.querySelector("#candidate-log-loss").textContent = data.calibrated_candidate.metrics.log_loss.toFixed(4);
    document.querySelector("#baseline-log-loss").textContent = data.home_rate_baseline.metrics.log_loss.toFixed(4);
    const decision = document.querySelector("#calibration-decision");
    decision.textContent = data.deployment.decision;
    decision.classList.toggle("good", data.deployment.decision === "deploy");
    document.querySelector("#calibration-note").textContent = data.deployment.reason;
  } catch {
    document.querySelector("#calibration-note").textContent = "Calibration audit is unavailable.";
  }
}

async function loadGames(gameDate) {
  renderLoading();
  try {
    const data = await fetchJson(`/api/v1/games?date=${encodeURIComponent(gameDate)}`);
    const games = [...data.games].sort((a, b) =>
      Math.max(b.home_win_probability, b.away_win_probability) - Math.max(a.home_win_probability, a.away_win_probability)
    );
    document.querySelector("#game-count").textContent = data.count;
    document.querySelector("#updated-label").textContent = new Date(`${data.date}T12:00:00`).toLocaleDateString([], { dateStyle: "long" });
    gamesRoot.innerHTML = games.length ? games.map(gameCard).join("") : "<p class='muted'>No games scheduled.</p>";
    gamesRoot.querySelectorAll(".game-card").forEach(card => {
      card.addEventListener("click", () => showGame(games.find(game => game.game_id === card.dataset.gameId)));
    });
  } catch (error) {
    gamesRoot.innerHTML = "";
    document.querySelector("#game-count").textContent = "0";
    notice.textContent = `${error.message}. Generate analysis for this date, then refresh.`;
    notice.classList.remove("hidden");
  }
}

function resultRow(game) {
  let comparison = '<span class="result-badge">Not tracked</span>';
  if (game.mlbai_tracked && !game.mlbai_verified) comparison = '<span class="result-badge">Awaiting settlement</span>';
  if (game.mlbai_verified) comparison = `<span class="result-badge ${game.mlbai_correct ? "correct" : "wrong"}">${game.mlbai_correct ? "✓ Correct" : "× Incorrect"} · ${escapeHtml(game.mlbai_lean)}</span>`;
  return `<tr>
    <td>${escapeHtml(new Date(`${game.official_date}T12:00:00`).toLocaleDateString([], { month: "short", day: "numeric" }))}</td>
    <td>${escapeHtml(game.away_team)} at ${escapeHtml(game.home_team)}</td>
    <td class="final-score">${escapeHtml(game.away_score)}–${escapeHtml(game.home_score)}</td>
    <td>${escapeHtml(game.winner_team)}</td>
    <td>${comparison}</td>
  </tr>`;
}

async function loadResults() {
  resultRows.innerHTML = '<tr><td colspan="5" class="muted">Loading season results…</td></tr>';
  const teamQuery = resultTeam ? `&team=${encodeURIComponent(resultTeam)}` : "";
  try {
    const data = await fetchJson(`/api/v1/results?season=2026&limit=${resultLimit}&offset=${resultOffset}${teamQuery}`);
    document.querySelector("#result-total").textContent = data.total;
    resultRows.innerHTML = data.results.length
      ? data.results.map(resultRow).join("")
      : '<tr><td colspan="5" class="muted">No completed games match this filter.</td></tr>';
    const first = data.total ? resultOffset + 1 : 0;
    const last = Math.min(resultOffset + data.returned, data.total);
    document.querySelector("#result-page").textContent = `${first}–${last} of ${data.total}`;
    document.querySelector("#newer-results").disabled = resultOffset === 0;
    document.querySelector("#older-results").disabled = resultOffset + data.returned >= data.total;
  } catch (error) {
    resultRows.innerHTML = `<tr><td colspan="5" class="muted">${escapeHtml(error.message)}</td></tr>`;
  }
}

dateInput.value = new URLSearchParams(window.location.search).get("date") || localToday();
dateInput.addEventListener("change", () => {
  history.replaceState(null, "", `?date=${dateInput.value}`);
  loadGames(dateInput.value);
});
document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
document.querySelector("#apply-team-filter").addEventListener("click", () => {
  resultTeam = document.querySelector("#team-filter").value.trim();
  resultOffset = 0;
  loadResults();
});
document.querySelector("#clear-team-filter").addEventListener("click", () => {
  document.querySelector("#team-filter").value = "";
  resultTeam = "";
  resultOffset = 0;
  loadResults();
});
document.querySelector("#newer-results").addEventListener("click", () => {
  resultOffset = Math.max(0, resultOffset - resultLimit);
  loadResults();
});
document.querySelector("#older-results").addEventListener("click", () => {
  resultOffset += resultLimit;
  loadResults();
});
document.querySelector("#team-filter").addEventListener("keydown", event => {
  if (event.key === "Enter") document.querySelector("#apply-team-filter").click();
});

loadPerformance();
loadSystemHealth();
loadCalibration();
loadGames(dateInput.value);
loadResults();
