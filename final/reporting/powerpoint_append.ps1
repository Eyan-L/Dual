$ErrorActionPreference='Stop'
$base='C:\Users\24165\AppData\Local\Temp\slack_trace_join_baseline_20261101'
$target='C:\Users\24165\Vault\Work\组会\组会_Slack_Serving_20261030.pptx'
$tag='dual_trace_join_baseline_26077464'
$app=[Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application')
$pres=$null
foreach($p in $app.Presentations){if($p.FullName -eq $target){$pres=$p;break}}
if($null -eq $pres){$pres=$app.Presentations.Open($target,0,0,-1)}
foreach($s in $pres.Slides){if($s.Tags.Item('SlackServingUpdate') -eq $tag){throw 'Native baseline result pages already present;no duplicate appended.'}}
$check=Get-Content -LiteralPath ($base+'\staging_check.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if($check.slides -ne 1 -or @($check.overflows).Count -ne 0){throw 'Staged result layout check failed.'}
function OriginalSnapshot($deck,$ids){
 $out=@()
 foreach($s in $deck.Slides){
  if($ids -notcontains [int]$s.SlideID){continue}
  $shapes=@()
  foreach($q in $s.Shapes){
   $t='';if($q.HasTextFrame -and $q.TextFrame.HasText){$t=$q.TextFrame.TextRange.Text}
   $shapes+=@{name=$q.Name;type=[int]$q.Type;x=[double]$q.Left;y=[double]$q.Top;w=[double]$q.Width;h=[double]$q.Height;text=$t}
  }
  $notes=''
  if($s.NotesPage.Shapes.Placeholders.Count -ge 2){$notes=$s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text}
  $out+=@{index=$s.SlideIndex;slideID=[int]$s.SlideID;shapes=$shapes;notes=$notes}
 }
 if($out.Count -ne $ids.Count){throw 'Original slide count differs.'}
 return ($out | ConvertTo-Json -Depth 12 -Compress)
}
$stage=$app.Presentations.Open($base+'\results_append_staging.pptx',-1,0,0)
if($stage.Slides.Count -ne 1){throw 'Expected exactly one prepared slide.'}
if([Math]::Abs($stage.PageSetup.SlideWidth-$pres.PageSetup.SlideWidth) -gt 1 -or [Math]::Abs($stage.PageSetup.SlideHeight-$pres.PageSetup.SlideHeight) -gt 1){throw 'Presentation dimensions differ.'}
$wasUnsaved=($pres.Saved -eq 0);$count=$pres.Slides.Count
$ids=@();foreach($s in $pres.Slides){$ids+=[int]$s.SlideID}
$before=OriginalSnapshot $pres $ids
$added=$pres.Slides.InsertFromFile($base+'\results_append_staging.pptx',$count,1,1)
if($added -ne 1 -or $pres.Slides.Count -ne $count+1){throw 'Unexpected slide insertion result.'}
if((OriginalSnapshot $pres $ids) -ne $before){throw 'Existing slide snapshot changed;do not save.'}
for($i=$count+1;$i -le $count+1;$i++){
 $s=$pres.Slides.Item($i)
 if($s.Tags.Item('SlackServingUpdate') -ne $tag){throw 'Imported result provenance tag missing.'}
 $s.Export($base+'\final_'+($i-$count)+'.png','PNG',1600,900)
}
$pres.Save()
@{target=$target;oldSlideCount=$count;addedSlides=1;newSlideCount=$pres.Slides.Count;originalSlidesUnchanged=$true;hadUnsavedUserEdits=$wasUnsaved;unsavedUserEditsPreserved=$true;saved=($pres.Saved -ne 0)} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath ($base+'\update_report.json') -Encoding UTF8
try{$stage.Close()}catch{Write-Output 'Result deck saved;temporary source close returned a COM cleanup error.'}
Write-Output ('APPENDED ONE RESULT SLIDE; TOTAL '+$pres.Slides.Count+'; ORIGINAL SLIDES PRESERVED')
