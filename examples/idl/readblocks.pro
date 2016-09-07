PRO getw_eh, event

   IF event.press EQ 4 THEN WIDGET_CONTROL, event.top, /DESTROY

   IF event.press NE 1 THEN RETURN

   WIDGET_CONTROL, event.top, GET_UVALUE = struct

   i    = struct.counter

   WIDGET_CONTROL, event.id, GET_UVALUE = win

   s    = event.x
   l    = event.y
   
;   print,'RAW s = ',s
;   print,'RAW l = ',l

   wset, win

   xyouts,s,l,strtrim(i,2),/device

   l = ((size(struct.buf))(2)-1) - l

   print,"point: ", i
   print,"   (IDL DATA l,s) = ", l, s
   print,"      (Fieldtype) = ", struct.fieldname
   print,"     (Data value) = ", struct.buf[s,l]

   struct.counter = struct.counter + 1

   WIDGET_CONTROL, event.top, SET_UVALUE = struct
END

PRO getw, w, h, base_id, win_id, buf, fieldname
   screen_sz = GET_SCREEN_SIZE()
   xscroll   = MIN([w,(screen_sz[0] < 1920) * 0.95])
   yscroll   = MIN([h,(screen_sz[1] < 1200) * 0.85])
   base_id = WIDGET_BASE(TITLE=fieldname,/COLUMN,XSIZE=w,YSIZE=h,X_SCROLL_SIZE=xscroll, $
              Y_SCROLL_SIZE=yscroll,/SCROLL)
   win = WIDGET_DRAW(base_id,SCR_XSIZE=w,SCR_YSIZE=h,RETAIN=2,/BUTTON_EVENTS,EVENT_PRO='getw_eh')
   WIDGET_CONTROL, base_id, /REALIZE
   WIDGET_CONTROL, win, GET_VALUE = win_id
   WIDGET_CONTROL, win, SET_UVALUE=win_id
   WIDGET_CONTROL, base_id, SET_UVALUE={counter:0L,buf:buf,fieldname:fieldname}
   XMANAGER,'getw',base_id,EVENT_HANDLER='getw_eh',/NO_BLOCK
END



pro readblocks
filename = dialog_pickfile(path='~/Desktop/MISRData')

status = mtk_file_to_gridlist(filename, gridcnt, gridlist)
print, 'Pick one'
for i = 0, gridcnt-1 do begin
    print, i, ') ', gridlist[i]
end
read,ans
gridname = gridlist[ans]

status = mtk_file_grid_to_fieldlist(filename, gridname, fieldcnt, fieldlist)
print, 'Pick one'
for i = 0, fieldcnt-1 do begin
    print, i, ') ', fieldlist[i]
end
read,ans
fieldname = fieldlist[ans]

status = mtk_file_grid_field_to_dimlist(filename, gridname, fieldname, ndim, dimlist, dimsize)
if (status eq 0) then begin
    print, "Enter dimension for ", fieldname
    for i = 0, ndim-1 do begin
        print, dimlist[i], "(0-", strtrim(dimsize[i]-1,2),")"
        read, ans
        fieldname = fieldname + "[" + strtrim(fix(ans),2) + "]"
    end
end else begin
    print, "This field doesn't have extra dimensions! That's ok!"
    print
endelse

status = mtk_file_to_blockrange(filename,sb,eb)
print,"Start block: ",strtrim(sb,2)
print,"End block:",strtrim(eb,2)

print,"Enter start block"
read,ans
sb = fix(ans)
print,"Enter end block"
read,ans
eb = fix(ans)

print,"Enter low-end clip data value"
read,ans
lcf = fix(ans)

print,"Enter high-end clip data value"
read,ans
hcf = fix(ans)

print,"Enter zoom amount (>0)"
read,ans
sf = fix(ans)
if (sf lt 1) then begin
    print, "zoom amount must be >0"
    stop
endif

print,"Enter data scale type (0=linear, 1=histogram equalization)"
read,ans
disptype = fix(ans)

print,filename
print,gridname,"/",fieldname
print,strtrim(sb,2),"-",strtrim(eb,2)

status = mtk_file_to_path(filename, path)

for blk = sb, eb do begin
    print, "Read block ", blk

    status = mtk_setregion_by_path_blockrange(path, blk, blk, region)

    status = mtk_readdata(filename, gridname, fieldname, region, buf)
    if (status ne 0) then stop

    bigbuf = rebin(buf,(size(buf))(1)*sf,(size(buf))(2)*sf,/sample)

    nbuf = bigbuf
    idx = where(nbuf le lcf, cnt)
    if (cnt ne 0) then nbuf[idx] = lcf

    idx = where(nbuf ge hcf, cnt)
    if (cnt ne 0) then nbuf[idx] = hcf

    if (blk eq sb) then begin
        bufstack = nbuf
    endif else begin
        bufstack = [[bufstack],[nbuf]]
    endelse
endfor 

getw, (size(bufstack))(1), (size(bufstack))(2), base_id, win_id, bufstack, fieldname

win = win_id

if (disptype eq 0) then begin
    tv,bytscl(bufstack),/order
endif else begin
    tv,hist_equal(bufstack),/order
endelse

print, "Minimum data value = ",strtrim(min(bufstack),2)
print, "Maximum data value = ",strtrim(max(bufstack),2)
help,buf

print, "Right-click in image window to close."
print, "Left-click to query coordinates in this data plane."

end
