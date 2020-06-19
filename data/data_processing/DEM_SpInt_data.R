#
## ACCESS PETREL DATA ##
#


# SETUP #


#CHECK, INSTALL AND LOAD REQUIRED PACKAGES
pkgs <- c("sp","rgdal","spdep","sf","rlist","elevatr","rlang","raster","spm")
for (pkg in pkgs) {
  if(pkg %in% rownames(installed.packages()) == FALSE) {install.packages(pkg)
    lapply(pkgs, require, character.only = TRUE)}
  else {
    lapply(pkgs, require, character.only = TRUE)}
}
rm(pkg,pkgs)
#SET WORKING DIRECTORY TO FILEPATH OF SCRIPT (PREFERRED DIRECTORY WHEN CLONING THE REPOSITORY)
#THIS WILL ONLY WORK WHEN USING R STUDIO, ELSE SET WD MANUALLY
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

#HELPER FUNCTIONS
matrify = function(raster_list) { #CONVERT RASTER LIST TO MATRIX LIST
  for (i in seq(1,length(raster_list))) {
    raster_list[[i]] = as.matrix(raster_list[[i]])
  }
  return(raster_list)
}

half_size = function(matrix_list) { #REDUCE MATRIX SIZE BY HALF BY DELETING EVERY SECOND ROW / COLUMN
  for (i in seq(1,length(matrix_list))) {
    matrix_list[[i]] = matrix_list[[i]][seq(1,nrow(matrix_list[[i]]),2),seq(1,ncol(matrix_list[[i]]),2)]
  }
  return(matrix_list)
}

# DATA #

#Access DTM data
# Data source: https://ndownloader.figshare.com/files/7446715
# File is located in "earthanalyticswk3/BLDR_LeeHill/pre-flood/lidar/pre_DTM_hill.tif"

dtm = raster("pre_DTM_hill.tif")
#Trim data
dtm = trim(dtm)
#Crop raster
dtm= dtm[7:(nrow(dtm)-8),
         (8 + 3*64):(ncol(dtm)-10), 
         drop=FALSE]

#Crop raster into 64 * 64 pixel tiles
a = aggregate(raster(dtm), n)
p = as(a, 'SpatialPolygons')
dtm_grid_raster = lapply(seq_along(p), function(i) crop(dtm, p[i]))
#Convert raster list to matrix list
dtm_grid_matrix = matrify(dtm_grid_raster)
dtm_grid_matrix_small = half_size(dtm_grid_matrix)
dtm_grid_matrix_small2 = half_size(dtm_grid_matrix_small) 
#Save matrix list as .JSON
#list.save(dtm_grid_matrix, 'list_dtm.json')
#list.save(dtm_grid_matrix_small, 'list_dtm_small.json')
#list.save(dtm_grid_matrix_small2, 'list_dtm_small2.json')