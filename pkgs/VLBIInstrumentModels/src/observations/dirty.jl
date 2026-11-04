export dirty_image, dirty_beam

"""
    dirty_image(t::VisTable, grid::AbstractRectiGrid)

The dirty image of the complex visibilities in the table `t` on the two-dimensional image
`grid`: the inverse Fourier transform of the measured visibilities, taking every other
visibility as zero. Needs VLBISkyModels to be loaded.
"""
function dirty_image end

"""
    dirty_beam(t::VisTable, grid::AbstractRectiGrid)

The dirty beam of the baseline coverage of the table `t` on the two-dimensional image `grid`:
the inverse Fourier transform of unit visibilities at the measured points and zero elsewhere.
Needs VLBISkyModels to be loaded.
"""
function dirty_beam end
