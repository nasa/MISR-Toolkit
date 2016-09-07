pro idl_test21

path = 37
sb = 50
eb = 60
status = mtk_path_blockrange_to_blockcorners(path, sb, eb, bc)
if (status ne 0) then begin 
    print,"Test Error: ", mtk_error_message(status)
    stop
end

print,status

help,/struct,bc

for i = bc.start_block, bc.end_block do begin
    pth = bc.path
    blk = bc.block[i].block_number
    ulc_lat = bc.block[i].ulc.lat
    ulc_lon = bc.block[i].ulc.lon
    ctr_lat = bc.block[i].ctr.lat
    ctr_lon = bc.block[i].ctr.lon
    lrc_lat = bc.block[i].lrc.lat
    lrc_lon = bc.block[i].lrc.lon

    status = mtk_dd_to_deg_min_sec(ulc_lat, ulc_lat_deg, ulc_lat_min, ulc_lat_sec)
    status = mtk_dd_to_deg_min_sec(ulc_lon, ulc_lon_deg, ulc_lon_min, ulc_lon_sec)
    status = mtk_dd_to_deg_min_sec(ctr_lat, ctr_lat_deg, ctr_lat_min, ctr_lat_sec)
    status = mtk_dd_to_deg_min_sec(ctr_lon, ctr_lon_deg, ctr_lon_min, ctr_lon_sec)
    status = mtk_dd_to_deg_min_sec(lrc_lat, lrc_lat_deg, lrc_lat_min, lrc_lat_sec)
    status = mtk_dd_to_deg_min_sec(lrc_lon, lrc_lon_deg, lrc_lon_min, lrc_lon_sec)


    print, format = '(%"Path: %3d  Block: %3d")', pth, blk
    print, format = '(%"   ULC: Lat %3d %02d %05.2f (%f)  Lon %4d %02d %05.2f (%f)")', $
      ulc_lat_deg, ulc_lat_min, ulc_lat_sec, ulc_lat, $
      ulc_lon_deg, ulc_lon_min, ulc_lon_sec, ulc_lon
    print, format = '(%"   CTR: Lat %3d %02d %05.2f (%f)  Lon %4d %02d %05.2f (%f)")', $
      ctr_lat_deg, ctr_lat_min, ctr_lat_sec, ctr_lat, $
      ctr_lon_deg, ctr_lon_min, ctr_lon_sec, ctr_lon
    print, format = '(%"   LRC: Lat %3d %02d %05.2f (%f)  Lon %4d %02d %05.2f (%f)")', $
      lrc_lat_deg, lrc_lat_min, lrc_lat_sec, lrc_lat, $
      lrc_lon_deg, lrc_lon_min, lrc_lon_sec, lrc_lon
endfor

end
