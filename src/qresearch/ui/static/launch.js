const $=id=>document.getElementById(id);
let CATALOG=null, DATASETS=[];
const PLAN_FIELDS=[["kind","choice",["rolling","expanding"]],["train","duration"],
 ["validation","duration"],["test","duration"],["purge","duration"],
 ["label_horizon","duration"],["warmup","duration"]];

function control(p, value){
  const v = value===undefined||value===null ? (p.default??"") : value;
  if(p.type==="choice") return `<select data-p="${p.name}">`+
    p.choices.map(c=>`<option ${String(v)===c?"selected":""}>${c}</option>`).join("")+`</select>`;
  if(p.type==="bool") return `<label class="count"><input type="checkbox" data-p="${p.name}" ${v?"checked":""}> ${p.name}</label>`;
  const t = (p.type==="int"||p.type==="float") ? "number" : "text";
  const step = p.type==="float" ? ' step="any"' : "";
  const ph = p.type==="duration" ? ' placeholder="PT5M"' : (p.type==="mapping"?' placeholder=\\'{"X":0.5}\\'':"");
  return `<label class="count">${p.name}</label><input type="${t}"${step}${ph} data-p="${p.name}" value="${v}" size="10">`;
}
function readParams(el, spec){
  const out={};
  for(const input of el.querySelectorAll("[data-p]")){
    const p=spec.params.find(x=>x.name===input.dataset.p); let v;
    if(input.type==="checkbox") v=input.checked;
    else if(input.value==="") continue;
    else if(p&&p.type==="int") v=parseInt(input.value,10);
    else if(p&&p.type==="float") v=parseFloat(input.value);
    else if(p&&(p.type==="mapping"||p.type==="list")){ try{v=JSON.parse(input.value);}catch(e){v=input.value;} }
    else v=input.value;
    out[input.dataset.p]=v;
  }
  return out;
}
function chip(group, specs, chosen, params){
  const div=document.createElement("div"); div.className="chip";
  const sel=`<select class="kind">`+specs.map(s=>
    `<option value="${s.kind}" ${s.kind===chosen?"selected":""}>${s.kind}</option>`).join("")+`</select>`;
  div.innerHTML=sel+`<span class="fields"></span>`+
    (group==="strategy"?"":`<button class="ghost rm">remove</button>`);
  const render=()=>{
    const spec=specs.find(s=>s.kind===div.querySelector(".kind").value);
    div.querySelector(".fields").innerHTML=spec.params.map(p=>control(p, params&&params[p.name])).join(" ");
    div._spec=spec;
  };
  div.querySelector(".kind").onchange=()=>{params=null;render();};
  const rm=div.querySelector(".rm"); if(rm) rm.onclick=()=>div.remove();
  render(); return div;
}
function collect(){
  const instruments=[...$("instruments").selectedOptions].map(o=>o.value);
  const cfg={
    dataset_id:$("dataset").value,
    features:[...$("features").children].map(d=>({kind:d.querySelector(".kind").value,params:readParams(d,d._spec)})),
    transforms:[...$("transforms").children].map(d=>({kind:d.querySelector(".kind").value,
      columns:(readParams(d,d._spec).columns)||[],params:(()=>{const p=readParams(d,d._spec);delete p.columns;return p;})()})),
    strategy:(()=>{const d=$("strategy").firstChild;return{kind:d.querySelector(".kind").value,params:readParams(d,d._spec)};})(),
    simulation:{initial_cash:parseFloat($("cash").value)},
    plan:Object.fromEntries([...$("plan").querySelectorAll("[data-p]")].filter(i=>i.value!=="").map(i=>[i.dataset.p,i.value])),
    cost_scenarios:[...document.querySelectorAll(".scen:checked")].map(c=>c.value),
    fill_rules:[...document.querySelectorAll(".fr:checked")].map(c=>c.value),
  };
  if(instruments.length) cfg.instrument_ids=instruments;
  if($("label").value) cfg.label=$("label").value;
  if($("experiment").value) cfg.experiment_id=$("experiment").value;
  return cfg;
}
async function post(url, body){
  const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  return {ok:r.ok, data:await r.json()};
}
function showErrors(data){
  const detail=(data.errors||[]).map(e=>`<li><code>${e.field}</code> — ${e.message}</li>`).join("");
  $("status").innerHTML=`<span class="err">${data.error||"invalid"}</span>`;
  $("previewout").innerHTML=detail?`<div class="card"><ul class="err">${detail}</ul></div>`:"";
}
async function doPreview(){
  $("status").textContent="checking…";
  const {ok,data}=await post("/api/preview",{config:collect()});
  if(!ok||data.error){showErrors(data);return null;}
  $("status").innerHTML=`<span class="ok">${data.estimate.runs} run(s) × `+
    `${data.estimate.folds_per_run} fold(s) = ${data.estimate.simulations} simulations</span>`;
  $("previewout").innerHTML=`<div class="card">${data.html}</div>`;
  $("yaml").textContent=data.yaml||"";
  return data;
}
$("preview").onclick=e=>{e.preventDefault();doPreview();};
$("launch").onclick=async e=>{
  e.preventDefault(); $("launch").disabled=true;
  const pv=await doPreview();
  if(!pv){$("launch").disabled=false;return;}
  const {ok,data}=await post("/api/backtests",{config:collect(),label:$("label").value||null});
  if(!ok){showErrors(data);$("launch").disabled=false;return;}
  location="/jobs/"+data.job_id;
};
$("addfeature").onclick=e=>{e.preventDefault();$("features").append(chip("features",CATALOG.features,CATALOG.features[0].kind));};
$("addtransform").onclick=e=>{e.preventDefault();$("transforms").append(chip("transforms",CATALOG.transforms,CATALOG.transforms[0].kind));};
$("dataset").onchange=()=>{
  const ds=DATASETS.find(d=>d.dataset_id===$("dataset").value); if(!ds)return;
  $("dsinfo").textContent=`${ds.bar_size} · ${ds.calendar_id} · ${ds.row_count.toLocaleString()} bars · `+
    `${ds.range_start.slice(0,10)} → ${ds.range_end.slice(0,10)}`+(ds.warning_count?` · ${ds.warning_count} warnings`:"");
  $("instruments").innerHTML=ds.instrument_ids.map(i=>`<option value="${i}">${i}</option>`).join("");
};
(async()=>{
  [CATALOG, DATASETS] = await Promise.all([
    fetch("/api/components").then(r=>r.json()), fetch("/api/datasets").then(r=>r.json())]);
  $("dataset").innerHTML=DATASETS.map(d=>`<option value="${d.dataset_id}">${d.dataset_id} (${d.asset_class})</option>`).join("");
  $("dataset").onchange();
  $("scenarios").innerHTML=["base","free","low","stressed"].map(s=>
    `<label class="count"><input type="checkbox" class="scen" value="${s}" ${s==="base"?"checked":""}> ${s}</label>`).join(" ");
  $("fillrules").innerHTML=["open_of_current_bar","next_open_after_eligibility"].map(s=>
    `<label class="count"><input type="checkbox" class="fr" value="${s}" checked> ${s}</label>`).join(" ");
  $("plan").innerHTML='<div class="row">'+PLAN_FIELDS.map(([n,t,c])=>
    t==="choice" ? `<label class="count">${n}</label><select data-p="${n}">`+c.map(o=>`<option>${o}</option>`).join("")+"</select>"
    : `<label class="count">${n}</label><input data-p="${n}" size="8" placeholder="PT1H" value="${DEFAULTS.plan[n]||""}">`
  ).join(" ")+"</div>";
  for(const f of DEFAULTS.features) $("features").append(chip("features",CATALOG.features,f.kind,f.params));
  $("strategy").append(chip("strategy",CATALOG.strategies,DEFAULTS.strategy.kind,DEFAULTS.strategy.params));
})();
