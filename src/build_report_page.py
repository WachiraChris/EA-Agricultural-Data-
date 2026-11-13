"""Build the self-contained HTML report page (reports/report_page.html)."""
import base64
import json
from common import ROOT

R = ROOT / "reports"
M = json.loads((R / "metrics.json").read_text())
G = json.loads((R / "demo_grid.json").read_text())
C, D = M["classifier"], M["data"]
T, U = C["test"], C["test_without_balancing"]


def img(name, alt):
    b = base64.b64encode((R / "figures" / f"{name}.png").read_bytes()).decode()
    return f'<figure class="fig"><div class="figscroll"><img alt="{alt}" src="data:image/png;base64,{b}"></div></figure>'


def pct(v):
    return f"{v * 100:.0f}%"


bands = "".join(
    f'<div class="brow"><span class="pill"><i class="dot {b["band"].lower().replace(" ", "")}"></i>{b["band"]}</span>'
    f'<div class="btrack"><div class="bfill" style="width:{b["observed_rate"] * 100 / 0.7:.1f}%"></div></div>'
    f'<span class="num">{pct(b["observed_rate"])}</span><span class="muted num">{b["n"]} cases</span></div>'
    for b in C["risk_bands"])
abl = "".join(
    f'<div class="brow abl"><span>{k}</span><div class="btrack"><div class="bfill" style="width:{(v - 0.5) / 0.4 * 100:.1f}%"></div></div>'
    f'<span class="num">{v:.2f}</span></div>' for k, v in C["ablation"].items())
cm = C["confusion"]
payload = json.dumps(dict(grid=G, bands=C["risk_bands"], thr=C["threshold"], q=M["regressor"]["resid_q"],
                          base=D["shortfall_rate_test"]), separators=(",", ":"))

html = f'''<title>East Africa Shortfall Advisor</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>
/* Layout: single reading column (report), with one wider working panel for the model demo. */
:root {{
  --bg:#f6f8f5; --surface:#ffffff; --ink:#131a15; --ink2:#4b554e; --muted:#7c877f; --line:#dde3dc;
  --accent:#1c6a49; --data:#2a78d6; --track:#e6ebe5; --fig:#fcfcfb;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --display:"Space Grotesk",system-ui,sans-serif; --body:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#101411; --surface:#181d19; --ink:#edf1ec; --ink2:#bcc5bd; --muted:#8a948c; --line:#2a312b;
  --accent:#5fc296; --data:#3987e5; --track:#262d27; --fig:#fcfcfb; color-scheme:dark }} }}
:root[data-theme="dark"] {{
  --bg:#101411; --surface:#181d19; --ink:#edf1ec; --ink2:#bcc5bd; --muted:#8a948c; --line:#2a312b;
  --accent:#5fc296; --data:#3987e5; --track:#262d27; --fig:#fcfcfb; color-scheme:dark }}
body {{ background:var(--bg); color:var(--ink); font:16px/1.6 var(--body); padding-block:40px 72px;
  padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); }}
main {{ max-width:860px; margin:0 auto; display:flex; flex-direction:column; gap:20px; }}
h1,h2,h3 {{ font-family:var(--display); text-wrap:balance; margin:0; line-height:1.2; }}
h1 {{ font-size:2.1rem; font-weight:600; }} h2 {{ font-size:1.4rem; margin-top:28px; }} h3 {{ font-size:1.05rem; margin-top:8px; }}
p,ul {{ margin:0; max-width:68ch; }} ul {{ padding-left:1.2em; display:flex; flex-direction:column; gap:6px; }}
.eyebrow {{ font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:var(--accent); font-weight:600; }}
.lede {{ font-size:1.1rem; color:var(--ink2); }} .muted {{ color:var(--muted); }} .small {{ font-size:.86rem; color:var(--ink2); }}
.num {{ font-variant-numeric:tabular-nums; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:0; border-block:1px solid var(--line); }}
.stat {{ padding:16px 16px 16px 0; }} .stat b {{ display:block; font:600 1.9rem/1.1 var(--display); }}
.stat span {{ font-size:.86rem; color:var(--ink2); }}
.panel {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:20px; display:flex; flex-direction:column; gap:16px; }}
.controls {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }}
label {{ display:flex; flex-direction:column; gap:6px; font-size:.86rem; color:var(--ink2); font-weight:500; }}
select,input[type=range] {{ font:inherit; width:100%; accent-color:var(--data); }}
select {{ padding:8px 10px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); }}
select:focus-visible,input:focus-visible {{ outline:2px solid var(--data); outline-offset:2px; }}
.out {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; }}
.out div b.pill {{ display:flex; }}
.chartscroll {{ overflow-x:auto; }} .chartscroll svg {{ min-width:520px; display:block; }}
.out div b {{ display:block; font:600 1.45rem/1.2 var(--display); }} .out div span {{ font-size:.82rem; color:var(--ink2); }}
.pill {{ display:inline-flex; align-items:center; gap:7px; font-weight:600; white-space:nowrap; }}
.dot {{ width:11px; height:11px; border-radius:50%; flex:none; background:var(--muted); }}
.dot.low {{ background:var(--good); }} .dot.moderate {{ background:var(--warn); }} .dot.high {{ background:var(--serious); }} .dot.veryhigh {{ background:var(--critical); }}
.verdict {{ border-left:3px solid var(--data); padding:4px 0 4px 12px; color:var(--ink2); max-width:none; }}
svg text {{ font:11px var(--body); fill:var(--muted); }}
.brow {{ display:grid; grid-template-columns:110px minmax(0,1fr) 44px 80px; gap:10px; align-items:center; font-size:.92rem; }}
.brow.abl {{ grid-template-columns:minmax(120px,210px) minmax(0,1fr) 44px; }}
.btrack {{ background:var(--track); border-radius:4px; height:14px; }} .bfill {{ background:var(--data); height:100%; border-radius:4px; }}
.tablewrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
th,td {{ text-align:left; padding:8px 12px 8px 0; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ font-weight:600; color:var(--ink2); font-size:.82rem; }} td.num,th.num {{ text-align:right; }}
.fig {{ margin:0; background:var(--fig); border:1px solid var(--line); border-radius:8px; padding:8px; }}
.figscroll {{ overflow-x:auto; }} .figscroll img {{ display:block; min-width:620px; width:100%; }}
.cm {{ display:grid; grid-template-columns:auto 1fr 1fr; gap:6px; max-width:460px; font-size:.86rem; align-items:center; }}
.cm .c {{ background:var(--track); border-radius:6px; padding:12px; text-align:center; }} .cm .c b {{ display:block; font:600 1.3rem var(--display); }}
.cm .h {{ color:var(--ink2); text-align:center; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:24px; }} .grid2>* {{ min-width:0; display:flex; flex-direction:column; gap:12px; }}
</style>
<main>
<header style="display:flex;flex-direction:column;gap:12px">
  <span class="eyebrow">Project report · November 2026 · prototype</span>
  <h1>East Africa crop shortfall advisor</h1>
  <p class="lede">A model that flags which crops, in which countries, are likely to yield more than 10% below their recent average. Built from an 8-record public-domain seed dataset, augmented with FAOSTAT yields and NASA POWER weather for 10 countries and 14 crops.</p>
</header>
<div class="stats">
  <div class="stat"><b class="num">{T["roc_auc"]:.2f}</b><span>ROC-AUC on unseen years (0.5 is chance)</span></div>
  <div class="stat"><b class="num">{pct(T["recall"])}</b><span>of real shortfalls caught, up from {pct(U["recall"])} before class balancing</span></div>
  <div class="stat"><b class="num">{pct(C["risk_bands"][3]["observed_rate"])} vs {pct(C["risk_bands"][0]["observed_rate"])}</b><span>shortfall rate in the Very high band against the Low band</span></div>
  <div class="stat"><b class="num">{D["rows"]:,}</b><span>country-crop-years after cleaning, 1981 to 2024</span></div>
</div>
<p>Two findings limit how it should be used. Most of the model's skill comes from recent yield history, with weather adding little at national scale. The companion yield estimate does not beat a "same as last year" guess.</p>

<h2>Try the model</h2>
<p class="small">Pick a country and crop and set the season's rainfall. Yield history is taken from FAOSTAT; the forecast is for the year after the last reported one. Values are pre-computed from the trained model.</p>
<section class="panel" aria-label="Model demo">
  <div class="controls">
    <label for="country">Country<select id="country"></select></label>
    <label for="crop">Crop<select id="crop"></select></label>
    <label for="rain">Season rainfall vs normal: <span id="rainv" class="num"></span><input id="rain" type="range" min="-50" max="50" step="10" value="0"></label>
  </div>
  <div class="out">
    <div><b id="oband" class="pill"></b><span>Shortfall risk band</span></div>
    <div><b id="oscore" class="num"></b><span>Risk score (flag at {C["threshold"]:.2f})</span></div>
    <div><b id="oyield" class="num"></b><span id="oyields"></span></div>
    <div><b id="olast" class="num"></b><span id="olasts"></span></div>
  </div>
  <p class="verdict" id="verdict"></p>
  <div>
    <h3 style="margin:0 0 4px">Risk score across rainfall scenarios</h3>
    <div class="chartscroll"><svg id="chart" viewBox="0 0 640 220" width="100%" role="img" aria-label="Risk score by rainfall scenario"></svg></div>
  </div>
</section>

<h2>Is the model working?</h2>
<p>The model was trained on {D["train_years"][0]} to {D["train_years"][1]} and tested on {D["test_years"][0]} to {D["test_years"][1]}, {D["test_rows"]:,} cases it never saw. {pct(D["shortfall_rate_test"])} of those were shortfalls.</p>
<div class="grid2">
  <div><h3>Share of cases in each risk band that ended in a shortfall</h3>{bands}
    <p class="small">Higher bands contain steadily more real shortfalls, which is the behaviour a warning tool needs.</p></div>
  <div><h3>Every test prediction, sorted by outcome</h3>
    <div class="cm"><span></span><span class="h">Predicted normal</span><span class="h">Predicted shortfall</span>
      <span class="h">Actual normal</span><div class="c"><b class="num">{cm["tn"]}</b>correct all-clear</div><div class="c"><b class="num">{cm["fp"]}</b>false alarm</div>
      <span class="h">Actual shortfall</span><div class="c"><b class="num">{cm["fn"]}</b>missed</div><div class="c"><b class="num">{cm["tp"]}</b>correct warning</div></div>
    <p class="small">{pct(T["precision"])} of warnings were right; {pct(T["recall"])} of shortfalls were caught.</p></div>
</div>
<h3>What class balancing changed</h3>
<p>Only {pct(D["shortfall_rate_train"])} of training rows were shortfalls ({D["class_counts_train"]["shortfall"]:,} against {D["class_counts_train"]["normal"]:,}). Left alone, every model learned to say "normal" almost always. Five balancing strategies were compared across three model families; logistic regression with random undersampling was selected.</p>
<div class="tablewrap"><table>
<tr><th></th><th class="num">Shortfalls caught</th><th class="num">Warnings right</th><th class="num">Balanced accuracy</th><th class="num">ROC-AUC</th></tr>
<tr><td>No balancing</td><td class="num">{pct(U["recall"])}</td><td class="num">{pct(U["precision"])}</td><td class="num">{pct(U["balanced_accuracy"])}</td><td class="num">{U["roc_auc"]:.2f}</td></tr>
<tr><td>With balancing (selected)</td><td class="num">{pct(T["recall"])}</td><td class="num">{pct(T["precision"])}</td><td class="num">{pct(T["balanced_accuracy"])}</td><td class="num">{T["roc_auc"]:.2f}</td></tr>
</table></div>
<h3>Where the skill comes from</h3>
<p class="small">ROC-AUC on test years when the model is given only part of the inputs. Bars start at 0.5 (chance).</p>
{abl}
<p>The model mainly detects crops already in a slump or with unstable yields. Weather alone has modest skill. National averages smooth out local droughts, so sub-national data is the main route to improvement.</p>

<h2>What the data showed</h2>
<h3>The seed dataset is 8 records</h3>
<p>Four crops, one tree, one rainfall zone, one soil type and one water feature, all free text and unverified. It was parsed into crop profiles and checked against modern yields; it could not be used for training.</p>
{img("02_seed_vs_modern_yield", "Seed yield ranges against modern FAOSTAT yields for sorghum, millet, cassava and cowpea")}
<h3>One variable at a time: imbalance and noise</h3>
<p>Crops are unevenly represented and shortfall years are the minority class. Cleaning removed 305 carried-forward repeat values and 9 implausible jumps.</p>
{img("05_uni_class_balance", "Rows per crop, rows per country and target class balance")}
<h3>Pairs of variables: dry seasons raise shortfalls</h3>
{img("08_bi_shortfall_by_rainfall", "Shortfall rate by main-season rainfall band")}
<h3>Many variables together: three climate groups</h3>
{img("13_multi_climate_pca", "Principal component analysis of country-year climate")}
{img("14_multi_crop_interactions", "Shortfall rate by crop and rainfall band, and by crop and country")}

<h2>How this can help</h2>
<ul>
<li><b>Early warning.</b> Official yield statistics arrive a year or more late. A risk band at the end of the season points attention to the country-crop pairs most likely to disappoint.</li>
<li><b>Advisory agents.</b> Tools built on the seed dataset gain a measured, testable signal alongside the historical notes.</li>
<li><b>Insurance and planning.</b> Each risk band has a known hit rate, which is the starting point for pricing or for triggering a review.</li>
</ul>
<h2>Limits</h2>
<ul>
<li>Predictions are national averages and say nothing about a district or farm.</li>
<li>About 36% of FAOSTAT yield values used are estimates, which are smoother than reality.</li>
<li>Country weather is the mean of 4 to 6 farming locations, not a cropland-weighted average.</li>
<li>The risk score is not a calibrated probability; read it through the bands.</li>
<li>The yield estimate ({M["regressor"]["test"]["model"]["mae_kg_ha"]:.0f} kg/ha average error) is worse than repeating last year's yield ({M["regressor"]["test"]["baseline_last_year"]["mae_kg_ha"]:.0f} kg/ha).</li>
<li>The seed records remain unverified against their cited sources.</li>
</ul>
<p class="small">Code, data, the full analysis and a Streamlit app are in the GitHub repository WachiraChris/EA-Agricultural-Data-.</p>
</main>
<script>
const P={payload};
const $=id=>document.getElementById(id), cases=P.grid.cases, rains=P.grid.rain;
const fmt=n=>Math.round(n).toLocaleString("en-US");
const bandOf=s=>s<0.35?0:s<0.5?1:s<0.65?2:3;
const countries=[...new Set(cases.map(c=>c.country))].sort();
$("country").innerHTML=countries.map(c=>`<option>${{c}}</option>`).join("");
$("country").value="Kenya";
function fillCrops(){{const cs=cases.filter(c=>c.country===$("country").value).map(c=>c.crop).sort();
  const prev=$("crop").value; $("crop").innerHTML=cs.map(c=>`<option>${{c}}</option>`).join("");
  $("crop").value=cs.includes(prev)?prev:(cs.includes("Maize")?"Maize":cs[0]);}}
function draw(k,i){{const W=640,H=220,l=40,r=14,t=14,b=38,x=j=>l+(W-l-r)*j/(rains.length-1),y=v=>t+(H-t-b)*(1-v);
  let s="";for(const v of [0,0.25,0.5,0.75,1])s+=`<line x1="${{l}}" x2="${{W-r}}" y1="${{y(v)}}" y2="${{y(v)}}" stroke="var(--line)" stroke-width="1"/><text x="${{l-8}}" y="${{y(v)+4}}" text-anchor="end">${{v.toFixed(2)}}</text>`;
  rains.forEach((rv,j)=>{{if(j%2===0||rains.length<8)s+=`<text x="${{x(j)}}" y="${{H-18}}" text-anchor="middle">${{rv>0?"+":""}}${{rv}}%</text>`}});
  s+=`<text x="${{(l+W-r)/2}}" y="${{H-3}}" text-anchor="middle">season rainfall vs normal</text>`;
  s+=`<line x1="${{l}}" x2="${{W-r}}" y1="${{y(P.thr)}}" y2="${{y(P.thr)}}" stroke="var(--muted)" stroke-dasharray="5 4" stroke-width="1"/><text x="${{W-r}}" y="${{y(P.thr)-5}}" text-anchor="end">flag threshold</text>`;
  s+=`<polyline fill="none" stroke="var(--data)" stroke-width="2" stroke-linejoin="round" points="${{k.risk.map((v,j)=>x(j)+","+y(v)).join(" ")}}"/>`;
  k.risk.forEach((v,j)=>{{s+=`<circle cx="${{x(j)}}" cy="${{y(v)}}" r="${{j===i?6:3.5}}" fill="var(--data)" stroke="var(--surface)" stroke-width="2"><title>Rainfall ${{rains[j]>0?"+":""}}${{rains[j]}}%: score ${{v.toFixed(2)}}</title></circle>`}});
  $("chart").innerHTML=s;}}
function update(){{const k=cases.find(c=>c.country===$("country").value&&c.crop===$("crop").value); if(!k)return;
  const rv=+$("rain").value,i=rains.indexOf(rv),risk=k.risk[i],ratio=k.ratio[i],B=P.bands[bandOf(risk)],exp=k.mean5*ratio;
  $("rainv").textContent=(rv>0?"+":"")+rv+"% ("+fmt(k.normal_rain*(1+rv/100))+" mm)";
  $("oband").innerHTML=`<i class="dot ${{B.band.toLowerCase().replace(" ","")}}"></i>${{B.band}}`;
  $("oscore").textContent=risk.toFixed(2);
  $("oyield").textContent=fmt(exp)+" kg/ha"; $("oyields").textContent="Estimated yield, "+k.year+" ("+((ratio-1)*100>=0?"+":"")+((ratio-1)*100).toFixed(1)+"% vs 5-year average)";
  $("olast").textContent=fmt(k.last)+" kg/ha"; $("olasts").textContent="Last reported yield, "+(k.year-1);
  $("verdict").textContent=`${{k.crop}} in ${{k.country}}: ${{B.band.toLowerCase()}} risk, ${{risk>=P.thr?"flagged as a likely shortfall":"not flagged"}}. In the test years, ${{Math.round(B.observed_rate*100)}}% of the ${{B.n}} cases in this band ended in a shortfall, against ${{Math.round(P.base*100)}}% overall. Likely yield range ${{fmt(exp*Math.exp(P.q[0]))}} to ${{fmt(exp*Math.exp(P.q[1]))}} kg/ha.`;
  draw(k,i);}}
$("country").addEventListener("change",()=>{{fillCrops();update()}}); $("crop").addEventListener("change",update); $("rain").addEventListener("input",update);
fillCrops(); update();
</script>
'''
(R / "report_page.html").write_text(html)
print(len(html) // 1024, "KB")
