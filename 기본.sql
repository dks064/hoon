--table list
select table_name from tabs

-- sampling
create table TBRLZM010_SAMPLE as
	select TRCR_NO from TBRLZM010
	group by TRCR_NO
	order by random(0,1)
	limit 200000;

-- export_view
create view EXPORT_VIEW as
	select * from TBRLZM010 T, TBRLZ010_SAMPLE S,
	where TRD_DTM > "20180101" || "00000000"
	and T.TRCR_NO = S.TRCR_NO

-- count 
select count(*) from TBRLZM010;

-- sample size
select 6964483/752909095*100 from dual;

select count(*) from TBRLZM010
where FRC_ID is not null;

commit;

# Deep Server
## Shell

```sh
echo "TABLE $1" >> $2.ctl
echo "FIELD TERMINATED BY ';' " >> $2.ctl
-- echo "FIELD TERMINATED BY ';' " >> $1.ctl
-- echo "OPTIONALLY ENCLOSED BY ';' " >> $1.ctl
``sh

sh export.sh ERPORT_VIEW TBRLZM010


source ~/py36/bin/activate
source deactivate

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=grudeep123

