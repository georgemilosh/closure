# Peppe Arrò	

def multistream_peppe(field,field_x,field_y,nrows=1,ncols=1,xx=(0,Lx),yy=(0,Ly),cmap="seismic",vmax=None,vmin=None,norm=None,levels=128,title=None,xlabel=None,ylabel=None,figname="prova.png",dpi=350,xnbins=10,ynbins=10,color="black",density=2,linewidth=0.5,arrowsize=0.5,xtext=0,ytext=0,pclose=True,xfigure=16,yfigure=9):
	
	import string
	alph=list(string.ascii_lowercase)
	
	fig,axes=plt.subplots(nrows=nrows,ncols=ncols,constrained_layout=True)
	fig.set_size_inches(xfigure,yfigure,forward=True)
	
	for it in range(nrows*ncols):
		
		axes.flatten()[it].contourf(x[round(xx[0]/x[1]):round(xx[1]/x[1])],y[round(yy[0]/y[1]):round(yy[1]/y[1])], \
		                            field[round(xx[0]/x[1]):round(xx[1]/x[1]),round(yy[0]/y[1]):round(yy[1]/y[1]),it].T, \
		                            cmap=cmap,vmax=vmax,vmin=vmin,norm=norm,levels=levels)
		
		axes.flatten()[it].streamplot(x[round(xx[0]/x[1]):round(xx[1]/x[1])],y[round(yy[0]/y[1]):round(yy[1]/y[1])], \
		                              field_x[round(xx[0]/x[1]):round(xx[1]/x[1]),round(yy[0]/y[1]):round(yy[1]/y[1]),it].T, \
		                              field_y[round(xx[0]/x[1]):round(xx[1]/x[1]),round(yy[0]/y[1]):round(yy[1]/y[1]),it].T, \
		                              color=color,linewidth=linewidth,density=density, arrowsize=arrowsize)
		
		axes.flatten()[it].set_aspect("equal",adjustable="box")
		axes.flatten()[it].set_xlim(xx[0],xx[1]-x[1])
		axes.flatten()[it].set_ylim(yy[0],yy[1]-y[1])  
		axes.flatten()[it].xaxis.set_major_locator(plt.MaxNLocator(xnbins))
		axes.flatten()[it].yaxis.set_major_locator(plt.MaxNLocator(ynbins))
		axes.flatten()[it].tick_params(axis="both",labelsize="x-large")
		axes.flatten()[it].text(xtext,ytext,r"$("+alph[it]+")\quad t="+str(t[it])+"\,\Omega_e^{-1}$",fontsize="x-large",bbox={"facecolor":"white","edgecolor":"black","fill":True})                                                        
	
	for n in axes:
		n[0].set_ylabel(ylabel,fontsize="x-large")
		
	for n in axes.T:
		n[-1].set_xlabel(xlabel,fontsize="x-large")	
	
	fig.suptitle(title,fontsize="xx-large")	
	
	mappable=mpl.cm.ScalarMappable(cmap=cmap)
	mappable.set_clim(vmin=vmin,vmax=vmax)
	fig.colorbar(mappable=mappable,ax=axes.flatten())
	
	if pclose:
		plt.savefig(figname,dpi=dpi)
		plt.close()

def streamplot_peppe(field,field_x,field_y,it=0,xx=(0,Lx),yy=(0,Ly),cmap="seismic",vmax=None,vmin=None,norm=None,levels=128,title=None,xlabel=None,ylabel=None,figname="prova.png",dpi=350,xnbins=10,ynbins=10,density=2,linewidth=0.5,arrowsize=0.5,color="black"):
	
	plt.contourf(x[round(xx[0]/x[1]):round(xx[1]/x[1])],y[round(yy[0]/y[1]):round(yy[1]/y[1])], \
	             field[round(xx[0]/x[1]):round(xx[1]/x[1]),round(yy[0]/y[1]):round(yy[1]/y[1]),it].T, \
	             cmap=cmap,vmax=vmax,vmin=vmin,norm=norm,levels=levels)
	
	plt.colorbar()             
	plt.gca().set_aspect("equal",adjustable="box")
	
	plt.streamplot(x[round(xx[0]/x[1]):round(xx[1]/x[1])],y[round(yy[0]/y[1]):round(yy[1]/y[1])], \
	               field_x[round(xx[0]/x[1]):round(xx[1]/x[1]),round(yy[0]/y[1]):round(yy[1]/y[1]),it].T, \
	               field_y[round(xx[0]/x[1]):round(xx[1]/x[1]),round(yy[0]/y[1]):round(yy[1]/y[1]),it].T, \
	               color=color,linewidth=linewidth,density=density, arrowsize=arrowsize)
	
	plt.xlim(xx[0],xx[1]-x[1])
	plt.ylim(yy[0],yy[1]-y[1])
	plt.locator_params(axis="x",nbins=xnbins)
	plt.locator_params(axis="y",nbins=ynbins)
	plt.xticks(fontsize="x-large")
	plt.yticks(fontsize="x-large")
	plt.xlabel(xlabel,fontsize="x-large")
	plt.ylabel(ylabel,fontsize="x-large")
	plt.title(title,fontsize="x-large")
	
	plt.tight_layout()
	
	plt.savefig(figname,dpi=dpi)
	plt.close()

def contour_peppe(field,psi=False,it=0,xx=(0,Lx),yy=(0,Ly),cmap="seismic",vmax=None,vmin=None,norm=None,levels=128,psi_levels=32,title=None,xlabel=None,ylabel=None,figname="prova.png",dpi=350,xnbins=10,ynbins=10,colors="black"):
	
	plt.contourf(x[round(xx[0]/x[1]):round(xx[1]/x[1])],y[round(yy[0]/y[1]):round(yy[1]/y[1])], \
	             field[round(xx[0]/x[1]):round(xx[1]/x[1]),round(yy[0]/y[1]):round(yy[1]/y[1]),it].T, \
	             cmap=cmap,vmax=vmax,vmin=vmin,norm=norm,levels=levels)
	
	plt.colorbar()             
	plt.gca().set_aspect("equal",adjustable="box")
	             
	if psi:
		plt.contour(x[round(xx[0]/x[1]):round(xx[1]/x[1])],y[round(yy[0]/y[1]):round(yy[1]/y[1])], \
		            Psi[round(xx[0]/x[1]):round(xx[1]/x[1]),round(yy[0]/y[1]):round(yy[1]/y[1]),0,it].T, \
		            colors=colors,linewidths=0.5,levels=psi_levels)
	
	plt.locator_params(axis="x",nbins=xnbins)
	plt.locator_params(axis="y",nbins=ynbins)
	plt.xticks(fontsize="x-large")
	plt.yticks(fontsize="x-large")
	plt.xlabel(xlabel,fontsize="x-large")
	plt.ylabel(ylabel,fontsize="x-large")
	plt.title(title,fontsize="x-large")
	
	plt.tight_layout()
	
	plt.savefig(figname,dpi=dpi)
	plt.close()                 

def save_virtual_satellite(vso,folder,dx=1.0,dy=1.0,dz=1.0,it=0):
	
	import pickle
	
	def pwrite(folder,field_name,field):
		f=open(folder+"/"+field_name+".dat","wb")
		pickle.dump(field,f)
		f.close()
	
	Bx_int=vso.interpolate(Bx[...,it],dx=dx,dy=dy,dz=dz)
	By_int=vso.interpolate(By[...,it],dx=dx,dy=dy,dz=dz)
	Bz_int=vso.interpolate(Bz[...,it],dx=dx,dy=dy,dz=dz)
	
	Ex_int=vso.interpolate(Ex[...,it],dx=dx,dy=dy,dz=dz)
	Ey_int=vso.interpolate(Ey[...,it],dx=dx,dy=dy,dz=dz)
	Ez_int=vso.interpolate(Ez[...,it],dx=dx,dy=dy,dz=dz)
	
	n_0_int=vso.interpolate(-rho_0[...,it],dx=dx,dy=dy,dz=dz)
	n_1_int=vso.interpolate(rho_1[...,it],dx=dx,dy=dy,dz=dz)
	
	ux_0_int=vso.interpolate((Jx_0/rho_0)[...,it],dx=dx,dy=dy,dz=dz)
	uy_0_int=vso.interpolate((Jy_0/rho_0)[...,it],dx=dx,dy=dy,dz=dz)
	uz_0_int=vso.interpolate((Jz_0/rho_0)[...,it],dx=dx,dy=dy,dz=dz)
	
	ux_1_int=vso.interpolate((Jx_1/rho_1)[...,it],dx=dx,dy=dy,dz=dz)
	uy_1_int=vso.interpolate((Jy_1/rho_1)[...,it],dx=dx,dy=dy,dz=dz)
	uz_1_int=vso.interpolate((Jz_1/rho_1)[...,it],dx=dx,dy=dy,dz=dz)
	
	pwrite(folder,"B",np.stack((Bx_int, By_int, Bz_int)))
	pwrite(folder,"E",np.stack((Ex_int, Ey_int, Ez_int)))
	pwrite(folder,"n_e",n_0_int)
	pwrite(folder,"n_i",n_1_int)
	pwrite(folder,"u_e",np.stack((ux_0_int, uy_0_int, uz_0_int)))
	pwrite(folder,"u_i",np.stack((ux_1_int, uy_1_int, uz_1_int)))
	pwrite(folder,"xyz",np.stack((vso.x_t, vso.y_t, vso.z_t)))
	pwrite(folder,"s",vso.s_t)	

def get_spectral_index(k,spec,N):
	
	from scipy.optimize import curve_fit
	
	def line(x,a,b):
		return a*x+b
	
	X=np.log10(k[1:])
	Y=np.log10(spec[1:])
	
	k_red=[]
	slopes=[]
	
	for i in range(len(k)//N):
		
		p,e=curve_fit(line,X[i*N:(i+1)*N],Y[i*N:(i+1)*N],sigma=Y[i*N:(i+1)*N])
		k_red.append(np.mean(k[i*N+1:(i+1)*N+1]))
		slopes.append(p[0])
	
	return np.array(k_red), np.array(slopes)

def get_PS_2D(species):
	
	if species=="electrons":
		
		uxx=np.gradient(Jx_0/rho_0,x,axis=0,edge_order=2)
		uxy=np.gradient(Jx_0/rho_0,y,axis=1,edge_order=2)
		uyx=np.gradient(Jy_0/rho_0,x,axis=0,edge_order=2)
		uyy=np.gradient(Jy_0/rho_0,y,axis=1,edge_order=2)
		uzx=np.gradient(Jz_0/rho_0,x,axis=0,edge_order=2)
		uzy=np.gradient(Jz_0/rho_0,y,axis=1,edge_order=2)
		
		P=(Pxx_0+Pyy_0+Pzz_0)/3
		theta=uxx+uyy
		
		PS=-Pxx_0*uxx-Pxy_0*uxy-Pxy_0*uyx-Pyy_0*uyy-Pxz_0*uzx-Pyz_0*uzy
		Ptheta=P*theta
		PiD=-(Pxx_0-P)*(uxx-theta/3)-(Pyy_0-P)*(uyy-theta/3)-(Pzz_0-P)*(-theta/3)-Pxy_0*(uyx+uxy)-Pxz_0*(uzx)-Pyz_0*(uzy)
	
	if species=="ions":	
		
		uxx=np.gradient(Jx_1/rho_1,x,axis=0,edge_order=2)
		uxy=np.gradient(Jx_1/rho_1,y,axis=1,edge_order=2)
		uyx=np.gradient(Jy_1/rho_1,x,axis=0,edge_order=2)
		uyy=np.gradient(Jy_1/rho_1,y,axis=1,edge_order=2)
		uzx=np.gradient(Jz_1/rho_1,x,axis=0,edge_order=2)
		uzy=np.gradient(Jz_1/rho_1,y,axis=1,edge_order=2)
		
		P=(Pxx_1+Pyy_1+Pzz_1)/3
		theta=uxx+uyy
		
		PS=-Pxx_1*uxx-Pxy_1*uxy-Pxy_1*uyx-Pyy_1*uyy-Pxz_1*uzx-Pyz_1*uzy
		Ptheta=P*theta
		PiD=-(Pxx_1-P)*(uxx-theta/3)-(Pyy_1-P)*(uyy-theta/3)-(Pzz_1-P)*(-theta/3)-Pxy_1*(uyx+uxy)-Pxz_1*(uzx)-Pyz_1*(uzy)
		
	return PS, Ptheta, PiD

def do_dot(fx,fy,fz,gx,gy,gz):
	return fx*gx+fy*gy+fz*gz
	
def do_cross(fx,fy,fz,gx,gy,gz):
	return fy*gz-fz*gy, fz*gx-fx*gz, fx*gy-fy*gx	

def get_T(species):
	
	bx=Bx/np.sqrt(Bx**2+By**2+Bz**2)
	by=By/np.sqrt(Bx**2+By**2+Bz**2)
	bz=Bz/np.sqrt(Bx**2+By**2+Bz**2)
	
	if species=="electrons":
		
		T=(Pxx_0+Pyy_0+Pzz_0)/(-3*rho_0)
		T_par=(Pxx_0*bx**2+Pyy_0*by**2+Pzz_0*bz**2+2*(Pxy_0*bx*by+Pxz_0*bx*bz+Pyz_0*by*bz))/(-rho_0)
		T_perp=(3*T-T_par)/2
	
	if species=="ions":
		
		T=(Pxx_1+Pyy_1+Pzz_1)/(3*rho_1)
		T_par=(Pxx_1*bx**2+Pyy_1*by**2+Pzz_1*bz**2+2*(Pxy_1*bx*by+Pxz_1*bx*bz+Pyz_1*by*bz))/(rho_1)
		T_perp=(3*T-T_par)/2
	
	return T, T_par, T_perp

def get_agyrotropy(species):
	
	bx=Bx/np.sqrt(Bx**2+By**2+Bz**2)
	by=By/np.sqrt(Bx**2+By**2+Bz**2)
	bz=Bz/np.sqrt(Bx**2+By**2+Bz**2)
	
	if species=="electrons":
		
		I1=Pxx_0+Pyy_0+Pzz_0
		I2=Pxx_0*Pyy_0+Pxx_0*Pzz_0+Pyy_0*Pzz_0-(Pxy_0**2+Pxz_0**2+Pyz_0**2)
		P_par=Pxx_0*bx**2+Pyy_0*by**2+Pzz_0*bz**2+2*(Pxy_0*bx*by+Pxz_0*bx*bz+Pyz_0*by*bz)
		
		
	if species=="ions":
		
		I1=Pxx_1+Pyy_1+Pzz_1
		I2=Pxx_1*Pyy_1+Pxx_1*Pzz_1+Pyy_1*Pzz_1-(Pxy_1**2+Pxz_1**2+Pyz_1**2)
		P_par=Pxx_1*bx**2+Pyy_1*by**2+Pzz_1*bz**2+2*(Pxy_1*bx*by+Pxz_1*bx*bz+Pyz_1*by*bz)	
	
	return 1-4*I2/((I1-P_par)*(I1+3*P_par))

def get_ExB():
	
	vDx,vDy,vDz=do_cross(Ex,Ey,Ez,Bx,By,Bz)/(Bx**2+By**2+Bz**2)
	
	return vDx, vDy, vDz

def get_Pdrift_2D():
	
	Fx=np.gradient(Pxx_0,x,axis=0,edge_order=2)+np.gradient(Pxy_0,y,axis=1,edge_order=2)
	Fy=np.gradient(Pxy_0,x,axis=0,edge_order=2)+np.gradient(Pyy_0,y,axis=1,edge_order=2)
	Fz=np.gradient(Pxz_0,x,axis=0,edge_order=2)+np.gradient(Pyz_0,y,axis=1,edge_order=2)
	
	vDx,vDy,vDz=get_ExB()
	vPx,vPy,vPz=do_cross(Bx,By,Bz,Fx,Fy,Fz)/(rho_0*(Bx**2+By**2+Bz**2))
	
	return vDx+vPx, vDy+vPy, vDz+vPz
	
def get_W(species):
	
	if species=="electrons":
		
		W=do_dot(Ex,Ey,Ez,Jx_0,Jy_0,Jz_0)
		
	if species=="ions":
		
		W=do_dot(Ex,Ey,Ez,Jx_1,Jy_1,Jz_1)	
	
	return W

def get_u(species):
	
	if species=="electrons":
		
		ux=Jx_0/rho_0
		uy=Jy_0/rho_0
		uz=Jz_0/rho_0
		
	if species=="ions":
		
		ux=Jx_1/rho_1
		uy=Jy_1/rho_1
		uz=Jz_1/rho_1	
	
	return ux, uy, uz

def get_uCM():
	
	ux_0,uy_0,uz_0=get_u("electrons")
	ux_1,uy_1,uz_1=get_u("ions")
	
	ux=((rho_0/qom_0)*ux_0+(rho_1/qom_1)*ux_1)/((rho_0/qom_0)+(rho_1/qom_1))
	uy=((rho_0/qom_0)*uy_0+(rho_1/qom_1)*uy_1)/((rho_0/qom_0)+(rho_1/qom_1))
	uz=((rho_0/qom_0)*uz_0+(rho_1/qom_1)*uz_1)/((rho_0/qom_0)+(rho_1/qom_1))
	
	return ux, uy, uz

def get_Jtot():
	return Jx_0+Jx_1, Jy_0+Jy_1, Jz_0+Jz_1

def get_D():
	
	Jx,Jy,Jz=get_Jtot()
	ux,uy,uz=get_u("electrons")
	uBx,uBy,uBz=do_cross(ux,uy,uz,Bx,By,Bz)
	
	return do_dot(Jx,Jy,Jz,Ex+uBx,Ey+uBy,Ez+uBz)-(rho_0+rho_1)*do_dot(ux,uy,uz,Ex,Ey,Ez)

def get_flux_2D(species):
		
	ux,uy,uz=get_u(species)	
	uBx,uBy,uBz=do_cross(ux,uy,uz,Bx,By,Bz)
		
	Fx=np.gradient(Ez+uBz,y,axis=1,edge_order=2)
	Fy=-np.gradient(Ez+uBz,x,axis=0,edge_order=2)
	Fz=np.gradient(Ey+uBy,x,axis=0,edge_order=2)-np.gradient(Ex+uBx,y,axis=1,edge_order=2)
	
	return Fx, Fy, Fz

def em_energy():
	return (Bx**2+By**2+Bz**2+Ex**2+Ey**2+Ez**2)/(8*np.pi)

def bulk_energy(species):
		
	if species=="electrons":
		
		E=(Jx_0**2+Jy_0**2+Jz_0**2)/(2*qom_0*rho_0)
		
	if species=="ions":
		
		E=(Jx_1**2+Jy_1**2+Jz_1**2)/(2*qom_1*rho_1)	
	
	return E
	
def thermal_energy(species):
	
	if species=="electrons":
		
		E=(Pxx_0+Pyy_0+Pzz_0)/2
		
	if species=="ions":
		
		E=(Pxx_1+Pyy_1+Pzz_1)/2	
	
	return E

def get_Ohm_MHD():
	
	ux,uy,uz=get_uCM()
	EMx,EMy,EMz=do_cross(ux,uy,uz,Bx,By,Bz)
	
	return -EMx, -EMy, -EMz

def get_Ohm_Hall_2D():
	
	Jx,Jy,Jz=get_Jtot()
	
	EHx,EHy,EHz=do_cross(Jx,Jy,Jz,Bx,By,Bz)/(-rho_0)
	
	EBx=np.gradient((Bx**2+By**2+Bz**2)/2,x,axis=0,edge_order=2)/(4*np.pi*rho_0)
	EBy=np.gradient((Bx**2+By**2+Bz**2)/2,y,axis=1,edge_order=2)/(4*np.pi*rho_0)
	
	ECx=(Bx*np.gradient(Bx,x,axis=0,edge_order=2)+By*np.gradient(Bx,y,axis=1,edge_order=2))/(-4*np.pi*rho_0)
	ECy=(Bx*np.gradient(By,x,axis=0,edge_order=2)+By*np.gradient(By,y,axis=1,edge_order=2))/(-4*np.pi*rho_0)
	ECz=(Bx*np.gradient(Bz,x,axis=0,edge_order=2)+By*np.gradient(Bz,y,axis=1,edge_order=2))/(-4*np.pi*rho_0)
	
	return EHx, EHy, EHz, EBx, EBy, ECx, ECy, ECz

def get_Ohm_P_2D():
	
	T=(Pxx_0+Pyy_0+Pzz_0)/(-3*rho_0)
	
	EPx=(np.gradient(Pxx_0,x,axis=0,edge_order=2)+np.gradient(Pxy_0,y,axis=1,edge_order=2))/rho_0
	EPy=(np.gradient(Pxy_0,x,axis=0,edge_order=2)+np.gradient(Pyy_0,y,axis=1,edge_order=2))/rho_0
	EPz=(np.gradient(Pxz_0,x,axis=0,edge_order=2)+np.gradient(Pyz_0,y,axis=1,edge_order=2))/rho_0
	
	ETx=-np.gradient(T,x,axis=0,edge_order=2)
	ETy=-np.gradient(T,y,axis=1,edge_order=2)
	
	Enx=-T*np.gradient(rho_0,x,axis=0,edge_order=2)/rho_0
	Eny=-T*np.gradient(rho_0,y,axis=1,edge_order=2)/rho_0
	
	return EPx, EPy, EPz, ETx, ETy, Enx, Eny

def Bp_coords(fx,fy):

	Bp=np.sqrt(Bx**2+By**2)
	
	return (fx*Bx+fy*By)/Bp, (fx*By-fy*Bx)/Bp
	
def B_coords(fx,fy,fz):
	
	Bp=np.sqrt(Bx**2+By**2)
	B=np.sqrt(Bx**2+By**2+Bz**2)
	
	return (fx*Bx+fy*By+fz*Bz)/B, (fx*By-fy*Bx)/Bp, (fx*Bx*Bz+fy*By*Bz-fz*Bp**2)/(Bp*B)
		
def get_MirrorT():
	
	T_0,T_0_par,T_0_perp=get_T("electrons")
	T_1,T_1_par,T_1_perp=get_T("ions")
	
	A_0=T_0_perp/T_0_par-1
	A_1=T_1_perp/T_1_par-1
	
	beta_0_par=-8*np.pi*rho_0*T_0_par/(Bx**2+By**2+Bz**2)
	beta_0_perp=-8*np.pi*rho_0*T_0_perp/(Bx**2+By**2+Bz**2)
	beta_1_par=8*np.pi*rho_1*T_1_par/(Bx**2+By**2+Bz**2)
	beta_1_perp=8*np.pi*rho_1*T_1_perp/(Bx**2+By**2+Bz**2)
	
	return beta_1_perp*A_1+beta_0_perp*A_0-1-0.5*(T_1_perp/T_1_par-T_0_perp/T_0_par)**2/(1/beta_1_par+1/beta_0_par)

def get_EKHI_speed(k):
	
	d=1/np.sqrt(4*np.pi*qom_0*rho_0)
	cAx=Bx/np.sqrt(4*np.pi*rho_0/qom_0)
	cAy=By/np.sqrt(4*np.pi*rho_0/qom_0)
	cAz=Bz/np.sqrt(4*np.pi*rho_0/qom_0)
	
	return cAx*k**2*d**2/(1+k**2*d**2), cAy*k**2*d**2/(1+k**2*d**2), cAz*k**2*d**2/(1+k**2*d**2),

def get_Az(dx,dy,Bx,By):
    
    Nx=Bx.shape[0]
    Ny=Bx.shape[1]
    Nz=Bx.shape[2]
    
    f=np.zeros((Nx,Ny,Nz))
    g=np.zeros((Nx,Ny,Nz))
    
    for iy in range(1,Ny):
        g[:,iy,:]=g[:,iy-1,:]+(Bx[:,iy-1,:]+Bx[:,iy,:])*dy/2
        
    for iy in range(0,Ny):
        for ix in range(1,Nx):
            f[ix,iy,:]=f[ix-1,iy,:]-(By[ix-1,0,:]+By[ix,0,:])*dx/2    

    return f+g
