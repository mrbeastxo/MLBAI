const gamesRoot = document.querySelector("#games");
const notice = document.querySelector("#notice");
const dateInput = document.querySelector("#game-date");
const dialog = document.querySelector("#game-dialog");
const dialogContent = document.querySelector("#dialog-content");

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
      <div class="factor-columns">
        <section><h4>Supporting the lean</h4>${factorList(game.strongest_supporting_factors, "No strong supporting factor.")}</section>
        <section><h4>Working against it</h4>${factorList(game.strongest_opposing_factors, "No strong opposing factor.")}</section>
      </div>
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
    const integrity = document.querySelector("#ledger-status");
    integrity.textContent = data.hash_chain_valid ? "Verified" : "Check failed";
    integrity.classList.toggle("good", data.hash_chain_valid);
  } catch {
    document.querySelector("#ledger-status").textContent = "Unavailable";
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

dateInput.value = new URLSearchParams(window.location.search).get("date") || localToday();
dateInput.addEventListener("change", () => {
  history.replaceState(null, "", `?date=${dateInput.value}`);
  loadGames(dateInput.value);
});
document.querySelector("#close-dialog").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });

loadPerformance();
loadGames(dateInput.value);
