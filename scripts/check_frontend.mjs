import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const root = new URL('../', import.meta.url);
const source = fs.readFileSync(new URL('frontend/app.js', root), 'utf8');
const html = fs.readFileSync(new URL('frontend/index.html', root), 'utf8');
const nodes = new Map();
const element = (s) => nodes.get(s) || nodes.set(s, {hidden:false,innerHTML:'',textContent:'',value:'',addEventListener(){},classList:{toggle(){}}}).get(s);
const report={mode:'synthetic fixture',rows:100,selected_model:'Test model',selected_version:'test-v1',models:[{name:'Test model',group:'all',metrics:{average_precision:.45,roc_auc:.72,brier:.08},training_seconds:.1}],holdout:{average_precision:.4,roc_auc:.7,brier:.09,recall:.5},splits:{train:{rows:55,start:0,end:54},test:{rows:15,start:85,end:99}},curves:{precision:[1,.5],recall:[0,1],calibration_predicted:[.1,.5],calibration_observed:[.12,.45]}};
const responses={'/api/report':report,'/api/examples':{examples:[]},'/api/graph':{nodes:[{id:'T1',type:'transaction'},{id:'card1:5',type:'entity'}],edges:[{source:'T1',target:'card1:5'}]},'/api/monitoring':{windows:[]},'/api/governance':{champion:'test-v1',history:[]},'/api/explanations':{status:'unavailable',reason:'No SHAP in test fixture'}};
const context=vm.createContext({console,Map,Set,Number,String,Math,JSON,Object,Array,Promise,FormData:class{},location:{hash:''},window:{addEventListener(){}},document:{querySelector:element,querySelectorAll:()=>[]},fetch:async p=>({ok:true,json:async()=>responses[p]})});
vm.runInContext(source,context);
await new Promise(r=>setTimeout(r,0));
assert.match(nodes.get('#mode').textContent,/SYNTHETIC/);
assert.match(nodes.get('#notice').innerHTML,/Synthetic data demo/);
for(const view of ['overview','laboratory','prediction','graph','monitoring','governance']) {
 vm.runInContext(`state.view='${view}'; render();`,context);
 const content=nodes.get('#view').innerHTML;
 assert(content.length>200,`${view} rendered`);
 assert(!content.includes('undefined'),`${view} has no undefined fields`);
 assert(!content.includes('NaN'),`${view} has no NaN values`);
}
assert.equal(vm.runInContext(`esc('<img onerror="bad">')`,context),'&lt;img onerror=&quot;bad&quot;&gt;');
assert.match(vm.runInContext(`predictionResult({probability:.75,threshold:.1,risk_category:'high',recommendation:'review',model_version:'test-v1'})`,context),/75.0%/);
for(const path of ['styles.css','app.js']) assert(fs.existsSync(new URL(`frontend/${path}`,root)));
assert(html.includes('aria-live="polite"'));
console.log('Frontend checks passed: six data-driven views, synthetic provenance, metric formatting, output escaping, assets.');
