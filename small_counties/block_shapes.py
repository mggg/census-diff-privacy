import pandas as pd
import geopandas as gpd
import maup
from local_tools import decennial_scraper

counties = {"bell": "027",
            "brazoria": "039",
            "cameron": "061",
            "galveston": "167",
            "nueces": "355"}

for cnty_name, cnty_fips in counties.items():
    blk_shapes = gpd.read_file("shapes/raw/tl_2020_48{0}_tabblock10/tl_2020_48{0}_tabblock10.shp".format(cnty_fips))
    blk_data = decennial_scraper.block_data_for_county("48", cnty_fips)
    blks = pd.merge(left=blk_shapes, right=blk_data, left_on="GEOID10", right_on="geoid")
    blks.to_file("shapes/blocks/{}_county_blocks_2010_data.shp".format(cnty_name))

for cnty_name, cnty_fips in counties.items():
    tract_shapes = gpd.read_file("shapes/raw/tl_2020_48{0}_tract10/tl_2020_48{0}_tract10.shp".format(cnty_fips))
    tract_data = decennial_scraper.tract_data_for_county("48", cnty_fips)
    tracts = pd.merge(left=tract_shapes, right=tract_data, left_on="GEOID10", right_on="geoid")
    tracts.to_file("shapes/tracts/{}_county_tract_2010_data.shp".format(cnty_name))



elects = pd.read_csv("../data/TX_statewide_18_Dem_Runofff.csv").drop(columns=["Unnamed: 0",'WVAP_pct', 
                                                                              'HVAP_pct', 'BVAP_pct',
                                                                              'non_WVAP_pct', 
                                                                              'non_HVAP_pct',
                                                                              'non_BVAP_pct'])
tx_prec_shapes = gpd.read_file("../data/TX_VTDs_cvap/TX_VTDs_cvap.shp")


for cnty_name, cnty_fips in counties.items():
    cnty_num = int(cnty_fips)
    prec_shapes = tx_prec_shapes.query("CNTY == @cnty_num")
    prec_shapes.to_file("shapes/precincts/{}_county_precincts.shp".format(cnty_name))


for cnty_name, cnty_fips in counties.items():
    prec_shapes = gpd.read_file("shapes/precincts/{}_county_precincts.shp".format(cnty_name))
    blk_shapes = gpd.read_file("shapes/blocks/{}_county_blocks_2010_data.shp".format(cnty_name))
    prec_shapes = prec_shapes.to_crs(blk_shapes.crs).set_index("CNTYVTD")
    assign = maup.assign(blk_shapes, prec_shapes)
    blk_shapes["CNTYVTD"] = assign
    blk_shapes.to_file("shapes/blocks/{}_county_blocks_2010_data.shp".format(cnty_name))





"""
    Galveston City
    City Boundaries were downloaded from the 
    (Texas Department of Transportation Open Data Portal)[https://gis-txdot.opendata.arcgis.com/search?tags=Boundaries]
    website.
"""

tx_city_boundaries = gpd.read_file("../../shapes/Texas/TxDOT_City_Boundaries/City.shp")
galveston_city = tx_city_boundaries.query("CITY_NM == 'Galveston'")

cnty_prec_shapes = gpd.read_file("shapes/precincts/galveston_county_precincts.shp")
cnty_blk_shapes = gpd.read_file("shapes/blocks/galveston_county_blocks_2010_data.shp")

galveston_city = galveston_city.to_crs(cnty_blk_shapes.crs)[["geometry"]]
cnty_prec_shapes = cnty_prec_shapes.to_crs(cnty_blk_shapes.crs)

city_blks = gpd.sjoin(left_df=cnty_blk_shapes, right_df=galveston_city, how="inner")
city_prec = gpd.sjoin(left_df=cnty_prec_shapes, right_df=galveston_city, how="inner")

city_blks.to_file("shapes/blocks/galveston_city_blocks_2010_data.shp")
city_prec.to_file("shapes/precincts/galveston_city_precincts.shp")

gpd.read_file("shapes/blocks/galveston_city_blocks_2010_data.shp").plot()
gpd.read_file("shapes/precincts/galveston_city_precincts.shp").plot()