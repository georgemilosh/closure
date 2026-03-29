# Peppe Arrò
# This script calculates the 1D power spectrum for both scalar and vector functions in 2D.

def scalar_spectrum_2D(field):
	
	# Repeated boundaries must be excluded according to the definition of the FFT.
	field_ft=np.fft.rfft2(field[0:-1,0:-1,0,:],axes=(0,1))
	
	# 2D power spectrum.
	spec_2D=(abs(field_ft)**2)/((nxc*nyc)**2)
	spec_2D[:,1:-1,:]*=2 # Some modes are doubled to take into account the redundant ones removed by numpy's rfft.
	kx=np.fft.fftfreq(nxc,x[1])*2*np.pi
	ky=np.fft.rfftfreq(nyc,y[1])*2*np.pi
	
	# The 1D magnetic field energy spectrum is calculated.
	spec_1D=np.zeros((nxc//2+1,len(t)))
	
	for iy in range(len(ky)):
		for ix in range(len(kx)):
			index=round( np.sqrt( (Lx*kx[ix]/(2*np.pi))**2+(Ly*ky[iy]/(2*np.pi))**2 ) )
			if index<=(nxc//2):
				spec_1D[index,:]+=spec_2D[ix,iy,:]
	
	return ky,spec_1D

def vector_spectrum_2D(field_x,field_y,field_z):
	
	# Repeated boundaries must be excluded according to the definition of the FFT.
	field_x_ft=np.fft.rfft2(field_x[0:-1,0:-1,0,:],axes=(0,1))
	field_y_ft=np.fft.rfft2(field_y[0:-1,0:-1,0,:],axes=(0,1))
	field_z_ft=np.fft.rfft2(field_z[0:-1,0:-1,0,:],axes=(0,1))
	
	# 2D power spectrum.
	spec_2D=(abs(field_x_ft)**2+abs(field_y_ft)**2+abs(field_z_ft)**2)/((nxc*nyc)**2)
	spec_2D[:,1:-1,:]*=2 # Some modes are doubled to take into account the redundant ones removed by numpy's rfft.
	kx=np.fft.fftfreq(nxc,x[1])*2*np.pi
	ky=np.fft.rfftfreq(nyc,y[1])*2*np.pi
	
	# The 1D magnetic field energy spectrum is calculated.
	spec_1D=np.zeros((nxc//2+1,len(t)))
	
	for iy in range(len(ky)):
		for ix in range(len(kx)):
			index=round( np.sqrt( (Lx*kx[ix]/(2*np.pi))**2+(Ly*ky[iy]/(2*np.pi))**2 ) )
			if index<=(nxc//2):
				spec_1D[index,:]+=spec_2D[ix,iy,:]
	
	return ky,spec_1D
