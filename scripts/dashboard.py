import json, html
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.common import COMPANY_NAMES, compute_ticker_statistics, load_score_history
except ImportError:
    from common import COMPANY_NAMES, compute_ticker_statistics, load_score_history

BASE_DIR = Path(__file__).resolve().parent
SCORES = BASE_DIR / "data" / "scores"
DOCS = BASE_DIR.parent / "docs"
TEMPLATE_PATH = BASE_DIR / "templates" / "dashboard.html"
CSS_TEMPLATE_PATH = BASE_DIR / "templates" / "dashboard.css"

# Authentic Bloomberg Terminal Color Palette
TERMINAL_PALETTE = [
    "#00e5ff",  # Terminal Cyan
    "#ff9900",  # Terminal Amber
    "#00ff66",  # Terminal Green
    "#ffff00",  # Terminal Yellow
    "#ff66cc",  # Terminal Magenta
    "#3399ff",  # Terminal Electric Blue
    "#ff3333",  # Terminal Red
    "#cc99ff",  # Terminal Lavender
    "#99ff99",  # Terminal Mint
    "#ffcc00"   # Terminal Gold
]

# PRESENTATION LAYER 1: Plotly Presenter

class PlotlyPresenter:
    """Serializes longitudinal statistics into Plotly.js chart payloads"""

    @staticmethod
    def _base_layout(xaxis_title: str, yaxis_title: str) -> dict:
        return {
            "paper_bgcolor": "#000000",
            "plot_bgcolor": "#000000",
            "font": {"family": "'IBM Plex Mono', 'Courier New', monospace", "color": "#888888", "size": 10},
            "xaxis": {
                "title": {"text": xaxis_title, "font": {"color": "#666666", "size": 10}},
                "gridcolor": "#171717",
                "zerolinecolor": "#262626",
                "tickfont": {"color": "#737373", "size": 10},
                "dtick": 1,
                "showline": True,
                "linecolor": "#222222"
            },
            "yaxis": {
                "title": {"text": yaxis_title, "font": {"color": "#666666", "size": 10}},
                "gridcolor": "#171717",
                "zerolinecolor": "#333333",
                "zerolinewidth": 1.2,
                "tickfont": {"color": "#737373", "size": 10},
                "showline": True,
                "linecolor": "#222222"
            },
            "legend": {
                "title": {"text": "<b>MONITORED SECURITIES</b>", "font": {"color": "#ff9900", "size": 9, "family": "'IBM Plex Mono', monospace"}},
                "font": {"color": "#cccccc", "size": 9.5, "family": "'IBM Plex Mono', monospace"},
                "bgcolor": "#0a0a0a",
                "bordercolor": "#222222",
                "borderwidth": 1,
                "orientation": "v"
            },
            "hovermode": "closest",
            "autosize": True,
            "margin": {"l": 55, "r": 25, "t": 20, "b": 40}
        }

    @classmethod
    def build_payloads(cls, stats: dict) -> tuple[list, dict, list, dict, list, dict]:
        """Build traces and layouts for Z-score, Raw Tone, and 2D Risk Dispersion"""
        tickers = sorted(stats.keys())
        traces_zscore = []
        traces_tone = []
        traces_dispersion = []

        for idx, ticker in enumerate(tickers):
            color = TERMINAL_PALETTE[idx % len(TERMINAL_PALETTE)]
            meta = stats[ticker]
            records = meta["records"]

            x_years = [r["year"] for r in records]
            y_zs = [r["z"] for r in records]
            y_tones = [r["tone"] for r in records]

            customdata_z = [
                [
                    f"{r['drift']:.3f}" if r["drift"] is not None else "N/A",
                    r["snippet"],
                    r["stock_return"],
                    f"{r['tone']:+.3f}",
                    f"{meta['mean']:+.3f}",
                    f"{meta['std']:.3f}"
                ]
                for r in records
            ]

            # Trace 1: Standardized Z-Score
            traces_zscore.append({
                "name": ticker,
                "x": x_years,
                "y": y_zs,
                "customdata": customdata_z,
                "mode": "lines+markers",
                "type": "scatter",
                "marker": {
                    "size": [11 if r["is_anomaly"] else 7 for r in records],
                    "color": [("#ff3333" if r["is_anomaly"] else color) for r in records],
                    "symbol": [("diamond" if r["is_anomaly"] else "square") for r in records],
                    "line": {"color": "#000000", "width": 1.5}
                },
                "line": {"color": color, "width": 1.5},
                "hovertemplate": (
                    "<b>[%{fullData.name}]</b> PERIOD: <b>%{x}</b><br>"
                    "------------------------------------<br>"
                    "TONE Z SCORE    : <b>%{y:+.2f}&sigma;</b><br>"
                    "RAW FINBERT TONE: <b>%{customdata[3]}</b><br>"
                    "HISTORICAL MEAN &mu;: %{customdata[4]} | &sigma;: %{customdata[5]}<br>"
                    "SBERT DRIFT     : <b>%{customdata[0]}</b><br>"
                    "EQUITY 1D RET   : <b>%{customdata[2]}</b><br>"
                    "------------------------------------<br>"
                    "EXCERPT: <i>\"%{customdata[1]}\"</i>"
                    "<extra></extra>"
                )
            })

            # Trace 2: Raw Polarity Tone
            traces_tone.append({
                "name": ticker,
                "x": x_years,
                "y": y_tones,
                "customdata": customdata_z,
                "mode": "lines+markers",
                "type": "scatter",
                "marker": {"size": 7, "color": color, "symbol": "square"},
                "line": {"color": color, "width": 1.5},
                "hovertemplate": (
                    "<b>[%{fullData.name}]</b> PERIOD: <b>%{x}</b><br>"
                    "------------------------------------<br>"
                    "RAW FINBERT TONE: <b>%{y:+.3f}</b><br>"
                    "TONE Z SCORE    : <b>%{customdata[3]}&sigma;</b><br>"
                    "SBERT DRIFT     : <b>%{customdata[0]}</b><br>"
                    "<extra></extra>"
                )
            })

            # Trace 3: 2D Dispersion
            valid_scatter = [r for r in records if r["drift"] is not None]
            if valid_scatter:
                traces_dispersion.append({
                    "name": ticker,
                    "x": [r["z"] for r in valid_scatter],
                    "y": [r["drift"] for r in valid_scatter],
                    "customdata": [[r["year"], r["snippet"], r["stock_return"]] for r in valid_scatter],
                    "mode": "markers+text",
                    "text": [f"{ticker} '{str(r['year'])[-2:]}" for r in valid_scatter],
                    "textposition": "top right",
                    "textfont": {"family": "'IBM Plex Mono', monospace", "size": 8, "color": color},
                    "type": "scatter",
                    "marker": {
                        "size": [12 if r["is_anomaly"] else 8 for r in valid_scatter],
                        "color": [("#ff3333" if r["is_anomaly"] else color) for r in valid_scatter],
                        "symbol": [("diamond" if r["is_anomaly"] else "circle") for r in valid_scatter],
                        "line": {"color": "#000000", "width": 1}
                    },
                    "hovertemplate": (
                        "<b>[%{fullData.name}]</b> PERIOD: <b>%{customdata[0]}</b><br>"
                        "------------------------------------<br>"
                        "TONE Z SCORE: <b>%{x:+.2f}&sigma;</b><br>"
                        "SBERT DRIFT : <b>%{y:.3f} &Delta;cos</b><br>"
                        "EQUITY RET  : <b>%{customdata[2]}</b><br>"
                        "<extra></extra>"
                    )
                })

        # Layout Z-Score
        layout_zscore = cls._base_layout("REGULATORY FISCAL FILING PERIOD", "STANDARDIZED TONE DEVIATION [ Z = (x - μ) / σ ]")
        layout_zscore["shapes"] = [
            {"type": "rect", "xref": "paper", "yref": "y", "x0": 0, "x1": 1, "y0": 1.5, "y1": 3.0,
             "fillcolor": "rgba(255, 51, 51, 0.08)", "line": {"width": 0}},
            {"type": "rect", "xref": "paper", "yref": "y", "x0": 0, "x1": 1, "y0": -3.0, "y1": -1.5,
             "fillcolor": "rgba(255, 51, 51, 0.08)", "line": {"width": 0}},
            {"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 1.5, "y1": 1.5,
             "line": {"color": "#ff3333", "width": 1.2, "dash": "dash"}},
            {"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": -1.5, "y1": -1.5,
             "line": {"color": "#ff3333", "width": 1.2, "dash": "dash"}},
            {"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 0, "y1": 0,
             "line": {"color": "#333333", "width": 1}}
        ]
        layout_zscore["annotations"] = [
            {"xref": "paper", "yref": "y", "x": 0.99, "y": 1.5, "text": "[UPPER TAIL BOUNDARY: +1.50σ]",
             "showarrow": False, "font": {"size": 9, "color": "#ff3333", "family": "'IBM Plex Mono', monospace"}, "xanchor": "right", "yanchor": "bottom"},
            {"xref": "paper", "yref": "y", "x": 0.99, "y": -1.5, "text": "[LOWER TAIL BOUNDARY: -1.50σ]",
             "showarrow": False, "font": {"size": 9, "color": "#ff3333", "family": "'IBM Plex Mono', monospace"}, "xanchor": "right", "yanchor": "top"}
        ]

        # Layout Raw Tone
        layout_tone = cls._base_layout("REGULATORY FISCAL FILING PERIOD", "RAW FINBERT POLARITY SCORE [ (Pos - Neg) / Total ]")

        # Layout 2D Dispersion
        layout_dispersion = cls._base_layout("STANDARDIZED TONE Z SCORE (σ)", "SBERT SEMANTIC DRIFT [ 1 - Cos(θ) ]")
        layout_dispersion["xaxis"]["dtick"] = None
        layout_dispersion["shapes"] = [
            {"type": "line", "x0": 1.5, "x1": 1.5, "y0": 0, "y1": 1.0, "line": {"color": "#ff3333", "width": 1, "dash": "dash"}},
            {"type": "line", "x0": -1.5, "x1": -1.5, "y0": 0, "y1": 1.0, "line": {"color": "#ff3333", "width": 1, "dash": "dash"}},
            {"type": "line", "x0": -3.0, "x1": 3.0, "y0": 0.5, "y1": 0.5, "line": {"color": "#ff3333", "width": 1, "dash": "dash"}}
        ]
        layout_dispersion["annotations"] = [
            {"x": 2.2, "y": 0.7, "text": "QUADRANT IV: COMPOUND TAIL RISK", "showarrow": False, "font": {"size": 8.5, "color": "#ff3333"}},
            {"x": 0.0, "y": 0.1, "text": "QUADRANT I: NOMINAL REGIME", "showarrow": False, "font": {"size": 8.5, "color": "#555555"}},
            {"x": 0.0, "y": 0.7, "text": "QUADRANT II: THEMATIC DRIFT SHIFT", "showarrow": False, "font": {"size": 8.5, "color": "#ffff00"}}
        ]

        return traces_zscore, layout_zscore, traces_tone, layout_tone, traces_dispersion, layout_dispersion


# PRESENTATION LAYER 2: Table and UI Component Renderer 
class TableRenderer:
    """Renders high density tabular and ribbon components for the Bloomberg UI"""

    @staticmethod
    def render_ticker_ribbon(stats: dict) -> str:
        """Format live ticker tape with latest score, z-score, and risk status"""
        items = []
        for ticker in sorted(stats.keys()):
            meta = stats[ticker]
            records = meta["records"]
            if not records:
                continue
            latest = records[-1]
            z_val = latest["z"]
            drift_str = f"&Delta;{latest['drift']:.3f}" if latest["drift"] is not None else "&Delta;--"

            if latest["is_anomaly"]:
                badge_html = f"<span class='tape-val alert'>{z_val:+.2f}&sigma; {drift_str} [BREACH]</span>"
            else:
                badge_html = f"<span class='tape-val ok'>{z_val:+.2f}&sigma; {drift_str} [NOMINAL]</span>"

            items.append(
                f"<div class='tape-item' onclick=\"isolateTicker('{ticker}')\" title='Click to isolate {ticker}'>"
                f"<span class='tape-ticker'>{ticker}</span>"
                f"{badge_html}"
                f"</div>"
            )
        return "".join(items)

    @staticmethod
    def render_exceedances_table(stats: dict) -> tuple[str, int]:
        """Generate high density HTML table for critical anomaly events"""
        events = []
        for ticker in sorted(stats.keys()):
            company = COMPANY_NAMES.get(ticker, ticker)
            for r in stats[ticker]["records"]:
                if r["is_anomaly"]:
                    events.append({
                        "ticker": ticker,
                        "company": company,
                        "year": r["year"],
                        "tone": r["tone"],
                        "z": r["z"],
                        "drift": f"{r['drift']:.3f}" if r["drift"] is not None else "N/A",
                        "stock_return": r["stock_return"],
                        "snippet": r["snippet"],
                        "full_snippet": r["full_snippet"]
                    })

        if not events:
            empty_html = """
            <div style="padding: 24px; text-align: center; color: var(--text-muted);">
              <strong style="color: var(--green); display: block; margin-bottom: 4px;">HISTORICAL INTEGRITY VERIFIED</strong>
              <span>ZERO REGULATORY 8K ANOMALY EXCEEDANCES DETECTED ACROSS MONITORED UNIVERSE (|z| &lt; 1.50&sigma;, &Delta; &lt; 0.50).</span>
            </div>
            """
            return empty_html, 0

        rows = []
        for ev in events:
            safe_full = ev["full_snippet"].replace("'", "&#39;").replace('"', '&quot;')
            rows.append(
                f"<tr>"
                f"<td><span class='ticker-code'>{html.escape(ev['ticker'])}</span></td>"
                f"<td>{html.escape(ev['company'])}</td>"
                f"<td class='num-col'>{ev['year']}</td>"
                f"<td class='num-col'>{ev['tone']:+.3f}</td>"
                f"<td class='num-col'><span class='tag-breach'>{ev['z']:+.2f}&sigma;</span></td>"
                f"<td class='num-col'>{ev['drift']}</td>"
                f"<td class='num-col'>{ev['stock_return']}</td>"
                f"<td><span class='tag-breach'>[CRITICAL_EXCEEDANCE]</span></td>"
                f"<td class='cell-snippet' onclick=\"openModal('{ev['ticker']}', '{ev['year']}', '{safe_full}')\" "
                f"title='Click to inspect full disclosure excerpt' style='cursor: pointer;'>"
                f"\"{html.escape(ev['snippet'])}\"</td>"
                f"</tr>"
            )

        table_html = f"""
        <table id="table-exceedances">
          <thead>
            <tr>
              <th>TICKER</th>
              <th>ISSUER NAME</th>
              <th class="num-col">PERIOD</th>
              <th class="num-col">RAW TONE</th>
              <th class="num-col">Z SCORE</th>
              <th class="num-col">SBERT DRIFT</th>
              <th class="num-col">EQUITY 1D</th>
              <th>REGIME STATUS</th>
              <th>SEC 8K DISCLOSURE EXCERPT (CLICK TO EXPAND)</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
        """
        return table_html, len(events)

    @staticmethod
    def render_universe_table(stats: dict) -> str:
        """Generate high density cross-sectional health table for all 10 monitored equities"""
        rows = []
        for ticker in sorted(stats.keys()):
            meta = stats[ticker]
            company = COMPANY_NAMES.get(ticker, ticker)
            records = meta["records"]
            latest = records[-1] if records else None

            latest_year = latest["year"] if latest else "N/A"
            latest_tone = f"{latest['tone']:+.3f}" if latest else "N/A"
            latest_z = f"{latest['z']:+.2f}&sigma;" if latest else "N/A"
            latest_drift = f"{latest['drift']:.3f}" if (latest and latest["drift"] is not None) else "N/A"

            if latest and latest["is_anomaly"]:
                status_html = "<span class='tag-breach'>[BREACH / EXCEEDANCE]</span>"
            elif meta["n"] >= 3:
                status_html = "<span class='tag-nominal'>[MONITORED_NOMINAL]</span>"
            else:
                status_html = "<span class='tag-expanding'>[EXPANDING_BASELINE]</span>"

            rows.append(
                f"<tr>"
                f"<td><span class='ticker-code'>{html.escape(ticker)}</span></td>"
                f"<td>{html.escape(company)}</td>"
                f"<td class='num-col'>{meta['n']}</td>"
                f"<td class='num-col'>{meta['mean']:+.3f}</td>"
                f"<td class='num-col'>{meta['std']:.3f}</td>"
                f"<td class='num-col'>{latest_year}</td>"
                f"<td class='num-col'>{latest_tone}</td>"
                f"<td class='num-col'>{latest_z}</td>"
                f"<td class='num-col'>{latest_drift}</td>"
                f"<td>{status_html}</td>"
                f"</tr>"
            )

        return f"""
        <table id="table-universe">
          <thead>
            <tr>
              <th>TICKER</th>
              <th>ISSUER NAME</th>
              <th class="num-col">SAMPLES (N)</th>
              <th class="num-col">LONGITUDINAL &mu;</th>
              <th class="num-col">VOLATILITY &sigma;</th>
              <th class="num-col">LATEST YR</th>
              <th class="num-col">LATEST TONE</th>
              <th class="num-col">LATEST Z</th>
              <th class="num-col">SBERT DRIFT</th>
              <th>MONITORING REGIME</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
        """

# PRESENTATION LAYER 3: Dashboard Compiler and Orchestration
def generate_dashboard(report: dict = None, history_data: dict = None) -> Path:
    """
    Generate an authentic, standalone Bloomberg/FactSet quantitative dashboard
    """
    if history_data is None:
        history_data = load_score_history(SCORES)

    # Domain Layer Delegation 
    stats = {ticker: compute_ticker_statistics(hist) for ticker, hist in history_data.items()}

    # Presentation Layer 1: Plotly Serialization
    (
        traces_zscore, layout_zscore,
        traces_tone, layout_tone,
        traces_dispersion, layout_dispersion
    ) = PlotlyPresenter.build_payloads(stats)

    # Presentation Layer 2: UI Component Rendering
    ticker_ribbon_html = TableRenderer.render_ticker_ribbon(stats)
    exceedances_table_html, total_anomalies = TableRenderer.render_exceedances_table(stats)
    universe_table_html = TableRenderer.render_universe_table(stats)

    all_tickers = sorted(stats.keys())
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    regime_label = "IDIOSYNCRATIC BREACH" if total_anomalies > 0 else "UNIFORM STABILITY"
    regime_color = "#ff3333" if total_anomalies > 0 else "#00ff66"
    alert_color = "#ff3333" if total_anomalies > 0 else "#00ff66"

    # Template Injection
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    css_content = CSS_TEMPLATE_PATH.read_text(encoding="utf-8")
    rendered_html = (
        template
        .replace("__DASHBOARD_CSS__", css_content)
        .replace("__TIMESTAMP__", timestamp)
        .replace("__TOTAL_TICKERS__", str(len(all_tickers)))
        .replace("__TOTAL_ANOMALIES__", str(total_anomalies))
        .replace("__ALERT_COLOR__", alert_color)
        .replace("__REGIME_LABEL__", regime_label)
        .replace("__REGIME_COLOR__", regime_color)
        .replace("__TICKER_RIBBON__", ticker_ribbon_html)
        .replace("__EXCEEDANCES_TABLE__", exceedances_table_html)
        .replace("__UNIVERSE_TABLE__", universe_table_html)
        .replace("__TRACES_ZSCORE_JSON__", json.dumps(traces_zscore))
        .replace("__LAYOUT_ZSCORE_JSON__", json.dumps(layout_zscore))
        .replace("__TRACES_TONE_JSON__", json.dumps(traces_tone))
        .replace("__LAYOUT_TONE_JSON__", json.dumps(layout_tone))
        .replace("__TRACES_DISPERSION_JSON__", json.dumps(traces_dispersion))
        .replace("__LAYOUT_DISPERSION_JSON__", json.dumps(layout_dispersion))
    )

    DOCS.mkdir(parents=True, exist_ok=True)
    index_path = DOCS / "index.html"
    index_path.write_text(rendered_html, encoding="utf-8")
    print(f"[SUCCESS] Institutional quantitative terminal generated at: {index_path}")
    return index_path

if __name__ == "__main__":
    generate_dashboard()
