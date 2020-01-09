/*===========================================================================
=                                                                           =
=                           MtkFileAttrList_test                            =
=                                                                           =
=============================================================================

                         Jet Propulsion Laboratory
                                   MISR
                               MISR Toolkit

            Copyright 2006, California Institute of Technology.
                           ALL RIGHTS RESERVED.
                 U.S. Government Sponsorship acknowledged.

============================================================================*/

#include "MisrFileQuery.h"
#include "MisrError.h"
#include <string.h>
#include <stdio.h>

int main () {

  MTKt_status status;           /* Return status */
  MTKt_boolean pass = MTK_TRUE; /* Test status */
  MTKt_boolean data_ok = MTK_TRUE; /* Data OK */
  int num_attrs;                /* Number of attributes */
  char **attrlist;              /* Attribute List */
  char filename[80];		/* HDF-EOS filename */
  int cn = 0;			/* Column number */
  int i;
  char *attrlist_expected[] = {"HDFEOSVersion", "StructMetadata.0",
                               "Path_number", "AGP_version_id",
                               "DID_version_id", "Number_blocks",
                               "Ocean_blocks_size", "Ocean_blocks.count",
			       "Ocean_blocks.numbers",
			       "SOM_parameters.som_ellipsoid.a",
			       "SOM_parameters.som_ellipsoid.e2",
			       "SOM_parameters.som_orbit.aprime",
			       "SOM_parameters.som_orbit.eprime",
			       "SOM_parameters.som_orbit.gama",
			       "SOM_parameters.som_orbit.nrev",
			       "SOM_parameters.som_orbit.ro",
			       "SOM_parameters.som_orbit.i",
			       "SOM_parameters.som_orbit.P2P1",
			       "SOM_parameters.som_orbit.lambda0",
			       "Origin_block.ulc.x",
			       "Origin_block.ulc.y",
			       "Origin_block.lrc.x",
			       "Origin_block.lrc.y",
			       "Start_block", "End block",
			       "Cam_mode", "Num_local_modes",
			       "Local_mode_site_name",
			       "Orbit_QA", "Camera", "coremetadata"};

  MTK_PRINT_STATUS(cn,"Testing MtkFileAttrList");

  /* Normal test call */
  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");

  status = MtkFileAttrList(filename, &num_attrs, &attrlist);
  if (status == MTK_SUCCESS)
  {
    if (num_attrs != sizeof(attrlist_expected) / sizeof(*attrlist_expected))
      data_ok = MTK_FALSE;

    for (i = 0; i < num_attrs; ++i)
      if (strcmp(attrlist[i],attrlist_expected[i]) != 0)
      {
        data_ok = MTK_FALSE;
	break;
      }
    MtkStringListFree(num_attrs, &attrlist);
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* File doesn't exists */
  strcpy(filename, "abcd.hdf");

  status = MtkFileAttrList(filename, &num_attrs, &attrlist);
  if (status == MTK_HDF_SDSTART_FAILED) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileAttrList(NULL, &num_attrs, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_GRP_ELLIPSOID_GM_P037_O029058_AA_F03_0024.hdf");
  status = MtkFileAttrList(filename, NULL, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileAttrList(filename, &num_attrs, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Normal test call */
  {
    char *attrlist_expected[] = {
      "Path_number",
      "AGP_version_id",
      "DID_version_id",
      "Number_blocks",
      "Ocean_blocks_size",
      "Ocean_blocks.count",
      "Ocean_blocks.numbers",
      "SOM_parameters.som_ellipsoid.a",
      "SOM_parameters.som_ellipsoid.e2",
      "SOM_parameters.som_orbit.aprime",
      "SOM_parameters.som_orbit.eprime",
      "SOM_parameters.som_orbit.gama",
      "SOM_parameters.som_orbit.nrev",
      "SOM_parameters.som_orbit.ro",
      "SOM_parameters.som_orbit.i",
      "SOM_parameters.som_orbit.P2P1",
      "SOM_parameters.som_orbit.lambda0",
      "Cam_mode",
      "Num_local_modes",
      "Local_mode_site_name",
      "Orbit_QA",
      "SOM_map_minimum_corner.x",
      "SOM_map_minimum_corner.y",
      "SOM_map_maximum_corner.x",
      "SOM_map_maximum_corner.y",
      "Start_block",
      "End_block",
      "Local_granule_id",
      "Local_version_id",
      "PGE_version",
      "Equator_crossing_time",
      "Equator_crossing_longitude",
      "Orbit_number",
      "Range_beginning_time",
      "Range_ending_time",
      "Software_version_information",
      "Software_version_tag",
      "Software_build_date",
      "Input_files",
      "Conventions",
      "title",
      "institution",
      "source",
      "history",
      "references",
      "config.deflate_level",
      "config.threads",
      "config.region_resolution",
      "config.eof_resolution",
      "config.domain_resolution",
      "config.num_tau_resid_gridpt",
      "config.tau_resid_gridpt",
      "config.ozone_coeff",
      "config.mu0_thresh",
      "config.region_topo_complex_thresh",
      "config.ae_rdqi1",
      "config.subr_topo_complex_thresh",
      "config.max_subr_avg_slope",
      "config.nsurf",
      "config.nascm",
      "config.nsdcm",
      "config.nrccm",
      "config.cloud_mask_decision_matrix(nsurf,nascm,nsdcm,nrccm)",
      "config.bright_thresh",
      "config.ae_rdqi3",
      "config.ae_rdqi4",
      "config.corr_mask_variance_limit",
      "config.reg_corr_mask_variance_limit",
      "config.ang_corr_thresh",
      "config.reg_ang_corr_thresh",
      "config.albedo_thresh_land",
      "config.albedo_thresh_water",
      "config.sigma_tau_default",
      "config.min_dw_subr_thresh",
      "config.min_dw_subr_thresh_17600",
      "config.min_dw_cam_thresh",
      "config.dw_band_mask",
      "config.chisq_abs_dw_rel_thresh_factor",
      "config.max_chisq_abs_dw_thresh",
      "config.max_chisq_geom_dw_thresh",
      "config.max_chisq_spec_dw_thresh",
      "config.max_chisq_maxdev_dw_thresh",
      "config.max_tau_unc_abs_thresh",
      "config.dw_tau_min_for_weights",
      "config.dw_tau_max_for_weights",
      "config.dw_chisq_weights(ncamera=9,nband=4)",
      "config.chisq_uncertainty_multiplier",
      "config.min_het_subr_thresh",
      "config.min_het_subr_thresh_regional",
      "config.min_het_subr_thresh_17600",
      "config.max_chisq_het_thresh",
      "config.combined_residual.max_chisq_het_thresh",
      "config.het_chisq_thresh_factor",
      "config.max_tau_unc_het_thresh",
      "config.eigenvector_variance_thresh",
      "config.max_het_tau_thresh",
      "config.glitter_thresh",
      "config.tau_ray_ref",
      "config.tau_ray_ref_550",
      "config.min_eq_refl_sigma",
      "config.min_tau",
      "config.max_tau",
      "config.windspeed_override",
      "config.use_windspeed_retrieval",
      "config.hdrf_shape_band_weight",
      "config.hdrf_shape_beta",
      "config.hdrf_shape_threshold_multiplier",
      "config.hdrf_shape_threshold",
      "config.smoothness.min_number_camera",
      "config.chisq_smooth_thresh",
      "config.ghost_correction_f(ncamera=9,nband=4)",
      "config.ghost_correction_window_height",
      "config.het_lambertian_require_bowl_shape",
      "config.het_lambertian_require_parabola_ok",
      "config.het_lambertian_disable_homog",
      "config.het_lambertian_single_iteration",
      "config.het_lambertian_allow_zero_optical_depth",
      "config.het_lambertian_normalize",
      "config.het_lambertian_disable_lambertian_term",
      "config.max_accept_tau_unc_per_het_case",
      "config.het_band_mask",
      "config.first_eigenvalue_for_eofs",
      "config.max_eofs",
      "config.homog_het_interface_tau0",
      "config.het_lambertian_max_bowl_shape_uncertainty",
      "config.het_lambertian_use_all_band_gridded_residual",
      "config.eof_disable",
      "config.het_lambertian_tau_upper_bound_disable",
      "config.het.use_aod_upper_bound_max",
      "config.het_bias_pixel_mode",
      "config.sim.enable",
      "config.sim.surface_type",
      "config.upper_bound.eq_refl_error",
      "config.underlight_albedo",
      "config.AerosolQualityScreener.min_cff",
      "config.AerosolQualityScreener.min_cff_3x3",
      "config.AerosolQualityScreener.min_confidence_index",
      "config.InverseChisqMetric.range_grid_size",
      "config.InverseChisqMetric.peak_interval_divisor",
      "config.combined_residual.use_het_relative_threshold",
      "config.GeographicExclusions.number_polygon",
      "config.GeographicExclusions.polygon.0",
      "config.GeographicExclusions.number_latitude",
      "config.GeographicExclusions.latitude.0",
    };

    strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");

    status = MtkFileAttrList(filename, &num_attrs, &attrlist);
    if (status == MTK_SUCCESS)
      {
        if (num_attrs != sizeof(attrlist_expected) / sizeof(*attrlist_expected))
          data_ok = MTK_FALSE;
        
        for (i = 0; i < num_attrs; ++i) {
          if (strcmp(attrlist[i],attrlist_expected[i]) != 0)
            {
              data_ok = MTK_FALSE;
              break;
            }
        }
        MtkStringListFree(num_attrs, &attrlist);
      }
  }

  if (status == MTK_SUCCESS && data_ok)
  {
    MTK_PRINT_STATUS(cn,".");
  }
  else
  {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  /* Argument Checks */
  status = MtkFileAttrList(NULL, &num_attrs, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  strcpy(filename, "../Mtk_testdata/in/MISR_AM1_AS_AEROSOL_P039_O002467_F13_23.b056-070.nc");
  status = MtkFileAttrList(filename, NULL, &attrlist);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  status = MtkFileAttrList(filename, &num_attrs, NULL);
  if (status == MTK_NULLPTR) {
    MTK_PRINT_STATUS(cn,".");
  } else {
    MTK_PRINT_STATUS(cn,"*");
    pass = MTK_FALSE;
  }

  if (pass) {
    MTK_PRINT_RESULT(cn,"Passed");
    return 0;
  } else {
    MTK_PRINT_RESULT(cn,"Failed");
    return 1;
  }
}
