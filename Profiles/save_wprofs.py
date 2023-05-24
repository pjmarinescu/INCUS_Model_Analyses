"""
Script to plot plan views from RAMS files

"""

import numpy as np
import datetime
import sys
import glob
import rams_tools
import os
import h5py
from collections import OrderedDict
import hdf5plugin
import copy
import pickle
from jug import TaskGenerator

print('start script')

tc_thr = 0.0001; # total condensate threshold
w_thr = 1.0; # vertical velocity threshold
use_depth = 1; #1 = save profiles of that contiguous depths of max_depth, 0 for grab all points that meet thresholds
max_depth = 2000.; # Maximum depth of contiguous featurse to include in analysis
gridnums = ['3','2','1']; # Loop through first, keep grid 3 first
npts = 200 ; #200 = 20km boundaries on G3

datainpath = '/nobackup/pmarines/PROD/'
savepath = '/nobackup/pmarines/Analyses/Profiles/Save/'

cases = 
svars = 

# Define bounding box function
minmax = lambda x: (np.min(x), np.max(x))
def bbox_extract_2D(datain,lat_2D,lon_2D, bbox):
    """
    Extract a sub-set of a dataset inside a lon, lat bounding box
    bbox=[lon_min lon_max lat_min lat_max].
    
    """
    # Check to see if there is need for a longitude adjustment
    if lon_2D[0,0] > 180:
        lons = lons - 360

    # Get indicies of lat lon box
    inregion = np.logical_and(np.logical_and(lon_2D > bbox[0],
                                             lon_2D < bbox[2]),
                              np.logical_and(lat_2D > bbox[1],
                                             lat_2D < bbox[3]))
    region_inds = np.where(inregion)
    #print(region_inds)
    imin, imax = minmax(region_inds[0])
    jmin, jmax = minmax(region_inds[1])
    return datain[:, imin:imax+1, jmin:jmax+1], lat_2D[imin:imax+1, jmin:jmax+1], lon_2D[imin:imax+1, jmin:jmax+1]

@TaskGenerator
def call_profile_analysis(path,savepath,svar_name,gridnum,tc_thr,w_thr,npts,use_depth,max_depth):
         
    print(path)
    # Get list of files with glob
    hefiles = sorted(glob.glob(path+'a-L*head.txt'))
    hfiles = sorted(glob.glob(path+'a-L*g'+gridnum+'.h5'))
    zcoords = np.array(rams_tools.calc_zcoords(hefiles[0]))
    dzcoords = np.diff(zcoords)
 
    print(hfiles)
 
    if gridnum == '3':
       rams_file = h5py.File(hfiles[0], 'r')
       lat = np.array(rams_file['GLAT'][:])
       lon = np.array(rams_file['GLON'][:])
       ny = np.shape(lat)[0]
       nx = np.shape(lat)[1]
       lon_bnds = [np.mean(lon[:,npts]), np.mean(lon[:,nx-npts])]
       lat_bnds = [np.mean(lat[npts,:]), np.mean(lat[ny-npts,:])]
       rams_file.close()
 
    print(lon_bnds)
    print(lat_bnds)
    # Convert lat_bnds and lon_bnds to format for bounding box function
    bbox = np.array([np.min(lon_bnds), np.min(lat_bnds), np.max(lon_bnds), np.max(lat_bnds)])
 
    # Create save_arr for pickle file to save
    save_w = OrderedDict()
    save_tc = OrderedDict()
    save_var = OrderedDict()
    date_list = []
    for f in np.arange(0,len(hfiles),5):
    #for f in np.arange(0,104,10): 
       cur_time = os.path.basename(hfiles[f])[4:21]
       print(cur_time)
 
       date_list = np.append(date_list,cur_time)
       # Read RAMS File
       rams_file = h5py.File(hfiles[f], 'r')
     
     # Load variables needed to calculate density
     #cp = 1004; rd = 287; p00 = 100000;
     #th = np.array(rams_file['THETA'][:])
     #pi = np.array(rams_file['PI'][:])
     #rv = np.array(rams_file['RV'][:])
 
     # Convert RAMS native variables to temperature and pressure
     #pres = np.power((pi/cp),cp/rd)*p00
     #temp = th*(pi/cp)
     #del(th,pi)
 
     #Calculate atmospheric density
     #dens = pres/(rd*temp*(1+0.61*rv))
     #del(pres,temp,rv)
 
       # Load RAMS Data
       lat = np.array(rams_file['GLAT'][:])
       lon = np.array(rams_file['GLON'][:])
       wp = np.array(rams_file['WP'][:])
 
       # Get data for specific lat/lon bounds 
       wp_crop,lat_new,lon_new = bbox_extract_2D(wp,lat,lon,bbox)
       del(wp)
       wpo = copy.deepcopy(wp_crop)
 
       #tc = np.array(rams_file['RTP'][:]-rams_file['RV'][:])*np.array(dens)
       tc = np.array(rams_file['RTP'][:]-rams_file['RV'][:])
       tc_crop,lat_new,lon_new = bbox_extract_2D(tc,lat,lon,bbox)
       del(tc)
 
       svar = np.array(rams_file[svar_name][:])
       svar_crop,lat_new,lon_new = bbox_extract_2D(tc,lat,lon,bbox)
       del(tc)
 
       wp_crop[tc_crop < tc_thr] = np.nan
       svar_crop[tc_crop < tc_thr] = np.nan
       tc_crop[tc_crop < tc_thr] = np.nan
       
       tc_crop[wp_crop < w_thr] = np.nan
       svar_crop[wp_crop < w_thr] = np.nan
       wp_crop[wp_crop < w_thr] = np.nan
     
       # Final datasets after thresholding
       tc_01 = ~np.isnan(tc_crop)
       wp_01 = ~np.isnan(wp_crop)
       svar_01 = ~np.isnan(svar_crop)
 
       if use_depth == 0:
 
          # Loop through saved column locations and save profiles of w
          save_w[f] = wp_01.flatten()
          save_tc[f] = tc_01.flatten()
          save_var[f] = svar_01.flatten()
 
       # Save location of pickle file with w profiles that meet criteria specified below
       savefile = savepath+casename+'_G'+str(int(gridnum))+'_'+str(tc_thr)+str(w_thr)+str(max_depth)+'_'+str(npts)+'_'+cur_time+'_'+saveadd+'.p'
 
       if use_depth == 1:
 
          # Sum boolean over vertical axis
          wp_sum_01 = np.nansum(wp_01,axis=0)
 
          # Find x,y points where it is worth doing contiguous calculations
          xid,yid = np.where(wp_sum_01 > max_depth/125.) # Quick check to see to find list of columns worth checking
 
          # Loop through columns that pass initial screening
          # So as to not loop over entire domain
          savex = []; savey = []
          for ii in np.arange(0,len(xid)):
             xi = xid[ii]; 
             yi = yid[ii];
   
             # Get current column of data
             vals = wp_01[:,xi,yi]
             #print(vals)
 
 	    # Find contiguous region indicies in each column
             runs = np.flatnonzero(np.diff(np.r_[np.int8(0),vals.view(np.int8), np.int8(0)])).reshape(-1, 2)
             runs[:, -1] -= 1
 
             # Find maximum contiguous region
             dz_max = 0
             for iii in np.arange(0,np.shape(runs)[0]):
                dz = zcoords[runs[iii,1]] - zcoords[runs[iii,0]]
                if dz > dz_max:
                   dz_max = dz
 
            # If maximum contiguous depth is greater than threshold, save column location
             if dz_max > max_depth:
                savex.append(xi)
                savey.append(yi)  
  
          # Loop through saved column locations and save profiles of w
          save_w[f] = np.zeros((len(zcoords),len(savex))) 
          save_tc[f] = np.zeros((len(zcoords),len(savex)))
          save_var[f] = np.zeros((len(zcoords),len(savex)))
          for ii in np.arange(0,len(savex)):
             save_w[f][:,ii] = wpo[:,savex[ii],savey[ii]]
             save_tc[f][:,ii] = tc_crop[:,savex[ii],savey[ii]]
             save_var[f][:,ii] = svar_crop[:,savex[ii],savey[ii]]
 
       # Save location of pickle file with w profiles that meet criteria specified below
       savefile = savepath+casename+'_G'+str(int(gridnum))+'_'+svar_name+'_'+str(tc_thr)+str(w_thr)+str(max_depth)+'_'+str(npts)+'_'+cur_time+'_'+saveadd+'.p'
 
    # Save Dictionary Object as a Pickle File
    with open(savefile, 'wb') as handle:
       pickle.dump([save_w,save_tc,save_var,date_list], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return ()

# Loop through cases
for c in np.arange(0,len(cases)):
    casename = cases[c]

    # Loop thought variables of interest
    for s in np.arange(0,len(svars)):
        svar_name = svars[s]

        # loop through grids
        for g in np.arange(0,len(gridnums)): 
        
           gridnum = gridnums[g]
        
           # Pathname to model data
           path = datainpath+casename+'/G3/out_30s/'
           #path = '/nobackup/ldgrant/'+casename+'/G3-poll/out30s/'
           
           # Call Analysis Function
           call_profile_analysis(path,savepath,svar_name,gridnum,tc_thr,w_thr,np:wts,use_depth,max_depth)