document.querySelectorAll('.cancel').forEach(b=>b.onclick=async e=>{
  e.preventDefault(); b.disabled=true;
  await fetch('/api/jobs/'+b.dataset.job,{method:'DELETE'}); location.reload();});
if(document.querySelector('.cancel')) setTimeout(()=>location.reload(), 2000);
