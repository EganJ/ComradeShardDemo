export dirty_image, dirty_beam

"""
    dirty_image(fov::Real, npix::Int, t::VisTable)

The dirty image of the complex visibilities in the table `t`, on a square grid with field of
view `fov` and `npix` pixels a side: the inverse Fourier transform of the measured
visibilities, taking every other visibility as zero. Needs VLBISkyModels to be loaded.
"""
function dirty_image end

"""
    dirty_beam(fov::Real, npix::Int, t::VisTable)

The dirty beam of the baseline coverage of the table `t`, on a square grid with field of view
`fov` and `npix` pixels a side: the inverse Fourier transform of unit visibilities at the
measured points and zero elsewhere. Needs VLBISkyModels to be loaded.
"""
function dirty_beam end
