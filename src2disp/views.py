# coding: UTF-8
import os
import io
import math
import asyncio
import numpy as np
import pandas as pd
import pyodide
from pyodide.http import open_url

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

global ax

def xyz2enu(center, v_xyz):
	"""
	XYZ座標でのベクトルv_xyzを
	centerを中心とするENU座標に変換
	"""
	f = 1.0/298.257223563 # WGS84
	rr = math.sqrt(center[0]**2 + center[1]**2)
	fm1 = 1.0/(1.0 - f)**2
	fai = math.atan2(fm1*center[2], rr)
	ram = math.atan2(center[1], center[0])
	
	rot = np.array(
		[[-math.sin(fai)*math.cos(ram), -math.sin(fai)*math.sin(ram), math.cos(fai)],
		 [-math.sin(ram)              ,  math.cos(ram)              , 0.0    ],
		 [ math.cos(fai)*math.cos(ram),  math.cos(fai)*math.sin(ram)     , math.sin(fai)]])

	neu = np.matmul(rot,v_xyz)
	enu = np.array([neu[1], neu[0], neu[2]])
	return enu

def llh2xyz(llh):
	"""
	lat,lon,hをXYZに変換
	"""
	lat = math.radians(llh[0])
	lon = math.radians(llh[1])
	a  = 6378.137          # earth semimajor axis in kilo-meters
	f  = 1.0/298.257223563  # reciprocal flattening
	e2 = (2.0 - f)*f     # eccentricity squared
	chi = math.sqrt((1.0 - e2*(math.sin(lat))**2))
	b = a*(1.0-e2)

	X = (a/chi + llh[2]) * math.cos(lat) * math.cos(lon)
	Y = (a/chi + llh[2]) * math.cos(lat) * math.sin(lon)
	Z = (b/chi + llh[2]) * math.sin(lat)

	return np.array([X, Y, Z])

def calc_dc3d(alpha, fault_para, site_list):
	'''
	DC3Dによる変位の計算
	
	site_listで指定した地点における
	fault_paraで指定した断層パラメータによる変位を計算
	
	各siteの変位結果はデータフレームでreturn
	'''
	
	# 結果格納用の空のリストを用意(このやり方が一番速い)
	a = []
	b = []
	c = []
	d = []
	e = []
	f = []
	g = []
	h = []
	hh = []
	
	# dc3d用にパラメータを変換
	strike = np.radians(90.0 - fault_para['Strike'])
	dip_slip = fault_para['slip-cm'] * np.sin(np.radians(fault_para['Rake']))
	strike_slip = fault_para['slip-cm'] * np.cos(np.radians(fault_para['Rake']))
	
	for i, site in site_list.iterrows(): # サイト毎のループ
		
		# 局所直交座標への変換
		xyz_site = llh2xyz((site['lat'], site['lon'], -site['dep']))
		xyz_source = llh2xyz((fault_para['Latitude'], fault_para['Longitude'], -fault_para['Depth']))
		delta = xyz_site - xyz_source
		enu = xyz2enu(xyz_source, delta)
		
		# DC3D用の座標（strike方向）
		d_strike = enu[0]*np.cos(strike) + enu[1]*np.sin(strike)
		d_dip = -enu[0]*np.sin(strike) + enu[1]*np.cos(strike)
		du = 0.0
		
		# DC3Dでの計算
		res = dc3d(alpha, d_strike, d_dip, du, fault_para['Depth'], fault_para['Dip'],
						-fault_para['length-km'], fault_para['length-km'], 
						-fault_para['width-km'], fault_para['width-km'], 
						strike_slip, dip_slip, 0.0)
		
		# 結果をENUに変換
		ue = res[0]*np.cos(strike) - res[1] * np.sin(strike)
		un = res[0]*np.sin(strike) + res[1] * np.cos(strike)
		uz = res[2]
		ur = np.sqrt(ue**2 + un**2)
		
		# 結果をリストに追加
		a.append(site['site'])
		b.append(site['lat'])
		c.append(site['lon'])
		d.append(site['dep'])
		e.append(ue)
		f.append(un)
		g.append(uz)
		h.append((ue**2.+un**2.)**0.5)
		hh.append(90.-math.degrees(math.atan2(un,ue)))
	
	# 結果をデータフレームに変換
	datalist = {'Site':a, 'Latitude':b, 'Longitude':c, 'Disp-cm':h, 'Direction':hh,
				 'E-ward-cm':e, 'N-ward-cm':f,'U-ward-cm':g}
	df_result = pd.DataFrame(data = datalist).set_index('Site')
	df_result = df_result.round(2)
	df_result = df_result.sort_values('Disp-cm', ascending=False)
	
	return df_result

def driver(alpha, dfsource, sites):
	global ax1, ax2
	
	fig1 = plt.figure(figsize=(8,8))
	fig2 = plt.figure(figsize=(8,8))
	ax1 = fig1.add_subplot(1,1,1)
	ax2 = fig2.add_subplot(1,1,1)
	lonrng1 = [121., 150.]
	latrng1 = [22., 48.]
	
	flon = dfsource.Longitude.values[0]
	flat = dfsource.Latitude.values[0]
	lo = round(flon)
	la = round(flat)
	lonrng2 = [lo-1., lo+1.]
	latrng2 = [la-1., la+1.]
	
	ax1.set_xlim(lonrng1)
	ax1.set_ylim(latrng1)
	ax2.set_xlim(lonrng2)
	ax2.set_ylim(latrng2)
	flt = ""
	
	# fault CMT
	loc = dfsource.loc[0,['Longitude','Latitude']]
	mec = dfsource.loc[0,['Strike','Dip','Rake']]
	#beach1 = beach(mec, xy=loc, width=0.2, linewidth=0.4, facecolor="darkmagenta", zorder=3)
	#ax.add_collection(beach1)
	ax1.plot(dfsource.Longitude, dfsource.Latitude, linestyle="None",
			c="darkmagenta", marker='*', markersize="25", zorder=3)
	ax2.plot(dfsource.Longitude, dfsource.Latitude, linestyle="None",
			c="darkmagenta", marker='*', markersize="25", zorder=3)
	
	ans = "done"
	flt += "  Length: %5.1f km, Width: %5.1f km, Slip: %5.1f cm" \
		  % (dfsource['length-km']*2.,dfsource['width-km']*2.,dfsource['slip-cm'])
	
	# grid deformation (for narrow scale)
	gridx = np.linspace(lonrng2[0], lonrng2[1], 21)[1:-1]
	gridy = np.linspace(latrng2[0], latrng2[1], 21)[1:-1]
	gx,gy = np.meshgrid(gridx,gridy)
	gridsite = pd.DataFrame({"lon":gx.flatten(), "lat":gy.flatten(), "dep":0., "site":"grid"})
	grdresult = calc_dc3d(alpha, dfsource.iloc[0,:], gridsite)
	scl = grdresult["Disp-cm"].values.max()
	x0 = grdresult["Longitude"].values
	y0 = grdresult["Latitude"].values
	dx = grdresult["E-ward-cm"].values / scl
	dy = grdresult["N-ward-cm"].values / scl
	ax2.quiver(x0, y0, dx, dy, units="inches", angles="xy", scale=1,
			  scale_units="inches", color="gray", zorder=4, width=0.02)
	ax2.set_title("Max arrow shows %4.1f cm disp." % scl)
	
	# site deformation
	result = calc_dc3d(alpha, dfsource.iloc[0,:], sites)
	x0 = result["Longitude"].values
	y0 = result["Latitude"].values
	dx = result["E-ward-cm"].values / scl
	dy = result["N-ward-cm"].values / scl
	ax1.quiver(x0, y0, dx, dy, units="inches", angles="xy", scale=1,
			  scale_units="inches", color="royalblue", zorder=5, width=0.07)
	ax2.quiver(x0, y0, dx, dy, units="inches", angles="xy", scale=1,
			  scale_units="inches", color="royalblue", zorder=5, width=0.07)
	df = result.sort_values('Disp-cm', ascending=True)
	
	# for plot figure
	display(dfsource, target="input", append=False)
	display(flt,target="flt", append=False)
	display(result,target="res", append=False)
	imgfl = './data/gebco_2023_n48.0_s22.0_w121.0_e150.0_cm.jpeg'
	async def imbyte(fl, fig1, fig2, lonrng1, latrng1, lonrng2, latrng2):
		global ax1, ax2
		iob = await ( await pyodide.http.pyfetch(fl) ).bytes()
		iobuf = io.BytesIO(iob)
		im = Image.open(iobuf)
		#print(im)
		ax1.imshow(im, extent=(lonrng1[0],lonrng1[1],latrng1[0],latrng1[1]), alpha=0.5)
		fig1.tight_layout()
		display(fig1, target="map1", append=False)
		#close-up
		ax2.imshow(im, extent=(lonrng1[0],lonrng1[1],latrng1[0],latrng1[1]), alpha=0.5)
		fig2.tight_layout()
		display(fig2, target="map2", append=False)
	
	loop = asyncio.get_event_loop()
	loop.run_until_complete(imbyte(imgfl, fig1, fig2, lonrng1, latrng1, lonrng2, latrng2))
	
	return
