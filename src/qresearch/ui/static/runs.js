const q=document.getElementById('q'),sc=document.getElementById('scenario'),
st=document.getElementById('strategy'),cmp=document.getElementById('cmp'),
clear=document.getElementById('clear'),shown=document.getElementById('shown');
const rows=[...document.querySelectorAll('#rows tr')];
function apply(){
  const t=q.value.toLowerCase(), s=sc.value, g=st.value; let n=0;
  for(const r of rows){
    const hay=r.dataset.hay.toLowerCase();
    const ok=(!t||hay.includes(t))&&(!s||hay.includes(s))&&(!g||hay.includes(g));
    r.style.display=ok?'':'none'; if(ok)n++;
  }
  shown.textContent=n+' shown';
}
function picked(){return [...document.querySelectorAll('.pick:checked')].map(c=>c.value);}
function sync(){cmp.disabled=picked().length<2;
  for(const r of rows) r.classList.toggle('sel', r.querySelector('.pick').checked);}
q.oninput=apply; sc.onchange=apply; st.onchange=apply;
document.getElementById('rows').addEventListener('change',sync);
document.getElementById('rows').addEventListener('click',e=>{
  if(e.target.tagName==='A'||e.target.classList.contains('pick'))return;
  const box=e.target.closest('tr').querySelector('.pick'); box.checked=!box.checked; sync();});
cmp.onclick=()=>location='/compare?runs='+picked().join(',');
clear.onclick=()=>{q.value='';sc.value='';st.value='';
  document.querySelectorAll('.pick').forEach(c=>c.checked=false);apply();sync();};
apply(); sync();
