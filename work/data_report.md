# Data report (Part A)

Seed 42. Split by well group, target 70/15/15 by page count.

## Pairing

- JSON files found: 51
- Paired: 51 (match type: {'exact': 51})
- JSON without image: 0
- Images without JSON: 1
  - `Datasets/images/60811406050000_608114060500_g31765 001 st00 bp00__master log_5640-28740_5 in_md(1).tif`

## Validation

- Bad pairs excluded: 0
- Valid pairs used for splitting: 51

## Splits

| split | pages | groups | 1-bit | grey | colour | very tall (h >= 143550) | median h | max h | median w |
|---|---|---|---|---|---|---|---|---|---|
| train | 35 (69%) | 22 | 9 | 0 | 26 | 9 | 50000 | 209000 | 1700 |
| val | 8 (16%) | 5 | 0 | 0 | 8 | 2 | 67200 | 212600 | 1700 |
| test | 8 (16%) | 5 | 0 | 0 | 8 | 2 | 44550 | 313800 | 1680 |

## Balance notes

- 1-bit pages all come from 1 group(s): `0000000009`. A group never crosses splits, so val/test can only hold 1-bit pages if that whole group goes there. Here it is in train, which means **val and test contain no 1-bit pages**; scores on them say nothing about 1-bit scans.
- No greyscale pages exist; every other page is colour.
- Very tall pages (top quartile by height) are spread across all three splits, but the single tallest page is in test, so test resizing/memory must handle it.
- Grouping by API well id merges sidetracks/bypasses (e.g. `6081140580` has 4 logs). Different wells in the same block (e.g. three GC743 wells, MC776-5 and MC776-6) stay separate groups, so some operator/style similarity across splits remains.
- Validation only reads the image header (size, mode); pixel data was not decoded here.

## Groups per split

- **train**: `0000000009` (9), `6080740286` (1), `6080840014` (1), `6081140410` (1), `6081140512` (1), `6081140535` (1), `6081140580` (4), `6081140757` (1), `6081140758` (1), `6081240048` (1), `6081240081` (1), `6081240102` (1), `6081741014` (2), `6081741170` (1), `6081741284` (1), `callaway 3-33 mudlog to td.pdf` (1), `critter creek 201-1807h_mud` (1), `rio_lavaca` (1), `rio_nueces` (2), `rio_san` (1), `shell_08852` (1), `shell_ac0336` (1)
- **val**: `6081140730` (1), `6081741142` (1), `6081840060` (1), `6081840065` (1), `mc765_p011` (4)
- **test**: `6081140605` (2), `6081741140` (3), `critter creek 202-1807h_mud` (1), `mudlog-trail mtn1palisades` (1), `mudlog_md` (1)

## All pages

| split | group | stem | mode | width | height |
|---|---|---|---|---|---|
| train | `0000000009` | 0000000009_d-582_0000020435-0001_log01_fe_1in_md | 1 | 4864 | 93524 |
| train | `0000000009` | 0000000009_d-582_0000020435-0001_log02_fe_5in_md | 1 | 4864 | 12035 |
| train | `0000000009` | 0000000009_d-582_0000020435-0001_log03_chromatograph_ratio | 1 | 4864 | 12031 |
| train | `0000000009` | 0000000009_d-582_0000020435-0001_log04_fe_5in_md | 1 | 4864 | 13041 |
| train | `0000000009` | 0000000009_d-582_0000020435-0001_log05_chromatograph_ratio | 1 | 4864 | 13051 |
| train | `0000000009` | 0000000009_d-582_0000020435-0001_log06_fe_5in_md | 1 | 4864 | 12050 |
| train | `0000000009` | 0000000009_d-582_0000020435-0001_log07_chromatograph_ratio | 1 | 4864 | 12031 |
| train | `0000000009` | 0000000009_d-582_0000020435-0001_log08_fe_5in_md | 1 | 4864 | 10007 |
| train | `0000000009` | 0000000009_d-582_0000020435-0001_log09_chromatograph_ratio | 1 | 4864 | 10000 |
| train | `6080740286` | 60807402860000_gb506#2_md_mudlog | RGB | 1536 | 9623 |
| train | `6080840014` | 60808400140000_hadrian 2_masterlog_2in_10000-21962 | RGB | 1700 | 50000 |
| train | `6081140410` | 60811404100000_gc743_dc141_st00bp00_mudlog_1in_md_9209_17052-rev | RGB | 1700 | 24200 |
| train | `6081140512` | 60811405120000_1inch tvd mudlog | RGB | 1535 | 15029 |
| train | `6081140535` | 60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400 | P | 2550 | 72600 |
| train | `6081140580` | 60811405800000_608114058000_g16786_001_00_00_md5_mud_show_2 | RGB | 1534 | 2600 |
| train | `6081140580` | 60811405800100_mdn-1_st00bp01_5 inch tvdmudlog_4560ft-22998ft | RGB | 1534 | 187600 |
| train | `6081140580` | 60811405800300_mdn-1_st01bp01_5 inch tvdmudlog_4560ft to 20251ft | RGB | 1275 | 120450 |
| train | `6081140580` | 60811405800500_608114058005_g16786_001_03_00_tvd5_mud | RGB | 1534 | 192600 |
| train | `6081140757` | 60811407570000_bp_gc743_15607_dc802_st00bp00_mudlog_5md_6900-17643 | RGB | 1700 | 110000 |
| train | `6081140758` | 608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD | P | 1700 | 26400 |
| train | `6081240048` | 60812400480000_ocs-g 26419 001 st00bp00_masterlog_7860-27400_5in_md_2 | RGB | 1533 | 198399 |
| train | `6081240081` | 60812400810000_ocs-g 16987 003 st00bp00_1inmd_mudlog_8921_15721 | RGB | 2200 | 18800 |
| train | `6081240102` | 60812401020000_xom_ocs-g_20351_ju103_st00bp00_mudlog_10200-30900ft_5in_tvd | RGB | 1700 | 209000 |
| train | `6081741014` | 60817410140000_mc822#10 1-1200 mud log | P | 1534 | 40000 |
| train | `6081741014` | 60817410140100_mc822#10bp01 1-240 tvd mud log | P | 1534 | 163220 |
| train | `6081741170` | 60817411700000_mc776-5_5in_md_combo_mud_log_20415_mms | RGB | 1536 | 150000 |
| train | `6081741284` | 60817412840000_mc_776_6_ml_mudlog_5757_23721_5in_md_final | RGB | 1552 | 182200 |
| train | `callaway 3-33 mudlog to td.pdf` | Callaway 3-33 mudlog to TD.PDF | RGB | 1275 | 56100 |
| train | `critter creek 201-1807h_mud` | Critter Creek 201-1807H_Mud_PDF_Vertical.pdf | RGB | 1275 | 44550 |
| train | `rio_lavaca` | Rio_Lavaca_1748_Unit_1_241_MUD | RGB | 1700 | 195800 |
| train | `rio_nueces` | Rio_Nueces_3744_Unit_1_212_MUD_5IN | RGB | 1700 | 121000 |
| train | `rio_nueces` | Rio_Nueces_3744_Unit_1_222_5IN_MUD | RGB | 1700 | 125400 |
| train | `rio_san` | Rio_San_Gabriel_2829_Unit_2_272_MUD | RGB | 1275 | 90750 |
| train | `shell_08852` | Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft | RGB | 1653 | 9356 |
| train | `shell_ac0336` | Shell_AC0336_OCS-G-36102_001_ST00BP01_5inchMD_MudLog_9100 ft -27770 ft | RGB | 1275 | 143550 |
| val | `6081140730` | 60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872 | RGB | 1653 | 28068 |
| val | `6081741142` | 60817411420000_mc_540_#1_st00_bp00_masterlog_3400-24195_5in_md | RGB | 1538 | 212600 |
| val | `6081840060` | 60818400600100_608184006001_g08037_002_01_00_5 inch mudlog_13452ft-19912ft | RGB | 1534 | 67200 |
| val | `6081840065` | 60818400650000_at0574_4_neptune_1inch_tvdmudlog_6300-17378 | RGB | 2307 | 36050 |
| val | `mc765_p011` | MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft | RGB | 1700 | 37400 |
| val | `mc765_p011` | MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft | RGB | 1700 | 176000 |
| val | `mc765_p011` | MC765_P011_ST00BP00_5inchTVD_Depth_MudLog_3686-16495ft | RGB | 1700 | 134200 |
| val | `mc765_p011` | MC765_P011_ST00BP01_5inchMD_Depth_MudLog_18230-21150ft | RGB | 1700 | 35200 |
| test | `6081140605` | 60811406050000_608114060500_g31765 001 st00 bp00__master log_5640-28740_5 in_md | RGB | 1680 | 238000 |
| test | `6081140605` | 60811406050100_ardennes_ocs-g31765 001 st00 bp01__master log_5640-36552_5 in_md | RGB | 1680 | 313800 |
| test | `6081741140` | 60817411400100_mc761_1_st1_mudlog_1md_17133_22886 | P | 1700 | 15400 |
| test | `6081741140` | 60817411400100_mc761_1_st1_mudlog_1tvd_17129_20968 | RGB | 1275 | 8250 |
| test | `6081741140` | 60817411400200_mc761_1_st1_bp01_mudlog_1md_17133_25775 | P | 1700 | 22000 |
| test | `critter creek 202-1807h_mud` | Critter Creek 202-1807H_Mud_Vertical_PDF.pdf | RGB | 1275 | 44550 |
| test | `mudlog-trail mtn1palisades` | Mudlog-Trail Mtn1Palisades | RGB | 1650 | 62325 |
| test | `mudlog_md` | Mudlog_MD_FREEMAN 3 24 ST | RGB | 1275 | 16500 |
