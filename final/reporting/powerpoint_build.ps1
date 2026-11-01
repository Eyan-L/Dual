$ErrorActionPreference='Stop'
$base='C:\Users\24165\AppData\Local\Temp\slack_serving_multiturn_native8_26075216'
$spec=Get-Content -LiteralPath ($base+'\slides.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$app=[Runtime.InteropServices.Marshal]::GetActiveObject('PowerPoint.Application')
function Color([string]$hex){$h=$hex.TrimStart('#');return [int]([Convert]::ToInt32($h.Substring(0,2),16)+256*[Convert]::ToInt32($h.Substring(2,2),16)+65536*[Convert]::ToInt32($h.Substring(4,2),16))}
$deck=$app.Presentations.Add(0)
$deck.PageSetup.SlideWidth=[single]$spec.width
$deck.PageSetup.SlideHeight=[single]$spec.height
$overflows=@()
$index=0
foreach($item in $spec.slides){
 $index++
 $slide=$deck.Slides.Add($index,12)
 $slide.FollowMasterBackground=0
 $slide.Background.Fill.Solid()
 $slide.Background.Fill.ForeColor.RGB=(Color '#FFFFFF')
 $slide.Tags.Add('SlackServingUpdate','flashinfer_multiturn_native8_26075216')
 $count=0
 foreach($o in $item.shapes){
  $count++
  if($o.type -eq 'text'){
   $shape=$slide.Shapes.AddTextbox(1,[single]$o.x,[single]$o.y,[single]$o.w,[single]$o.h)
   $shape.Name=('result_text_'+$index+'_'+$count)
   $frame=$shape.TextFrame
   $frame.AutoSize=0;$frame.WordWrap=-1
   $frame.MarginLeft=0;$frame.MarginRight=0;$frame.MarginTop=0;$frame.MarginBottom=0
   $frame.TextRange.Text=([string]$o.text).Replace("`n","`r")
   $range=$frame.TextRange
   $range.Font.Name='微软雅黑';$range.Font.Size=[single]$o.size
   if($o.bold){$range.Font.Bold=-1}else{$range.Font.Bold=0}
   $range.Font.Color.RGB=(Color $o.color)
   $range.ParagraphFormat.Alignment=[int]$o.align
   $range.ParagraphFormat.Bullet.Visible=0
   if($range.BoundHeight -gt ([single]$o.h+2)){$overflows+=@{slide=$index;shape=$shape.Name;text=$o.text;height=$range.BoundHeight;boxHeight=$o.h}}
  }elseif($o.type -eq 'rect'){
   $shape=$slide.Shapes.AddShape(1,[single]$o.x,[single]$o.y,[single]$o.w,[single]$o.h)
   $shape.Fill.Solid();$shape.Fill.ForeColor.RGB=(Color $o.color);$shape.Line.Visible=0
  }elseif($o.type -eq 'line'){
   $shape=$slide.Shapes.AddLine([single]$o.x,[single]$o.y,[single]$o.x2,[single]$o.y2)
   $shape.Line.ForeColor.RGB=(Color $o.color);$shape.Line.Weight=[single]$o.width
  }elseif($o.type -eq 'image'){
   $shape=$slide.Shapes.AddPicture([string]$o.path,0,-1,[single]$o.x,[single]$o.y,[single]$o.w,[single]$o.h)
  }else{throw ('Unknown shape type '+$o.type)}
 }
 $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=[string]$item.notes
 Write-Output ('STAGED_SLIDE '+$index+' '+$item.title)
}
$deck.SaveAs($base+'\results_append_staging.pptx',24)
for($i=1;$i -le $deck.Slides.Count;$i++){$deck.Slides.Item($i).Export($base+'\preview_'+$i+'.png','PNG',1600,900)}
$report=@{slides=$deck.Slides.Count;overflows=$overflows;staging=$base+'\results_append_staging.pptx'}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath ($base+'\staging_check.json') -Encoding UTF8
$deck.Close()
Write-Output ('STAGED '+$index+' SLIDES; OVERFLOWS '+$overflows.Count)
