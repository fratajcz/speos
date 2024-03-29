
from speos.utils.config import Config
from speos.visualization.diagnosticwrapper import GraphDiagnosticWrapper

import os
from speos.visualization.settings import *
import matplotlib.pyplot as plt

config = Config()
config.logging.dir = "speos/tests/logs/"

config.name = "DiagnosticTest"
config.model.save_dir = "speos/tests/models/"
config.inference.save_dir = "speos/tests/results"
config.model.plot_dir = "speos/tests/plots"
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation", "cardiovascular_disease", "insulin_disorder", "monogenic_diabetes", "bodymass_disorder"], adjacency_tag="")

"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation", "cardiovascular_disease"], adjacency_tag="hetionet_regulates")
fig, ax = diagnostic.get_diagnostics("degrees", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_degrees_regulates.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation", "cardiovascular_disease"], adjacency_tag="hetionet_regulates")
fig, ax = diagnostic.get_diagnostics("paths", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_paths_regulates.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="GRNDB")
fig, ax = diagnostic.get_diagnostics("paths", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_paths_grndb.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag=["GRNDB-adipose_tissue", "blood"])
fig, ax = diagnostic.get_diagnostics("paths", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_paths_grndb_adipose.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="GRNDB")
fig, ax = diagnostic.get_diagnostics("degrees", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_degrees_grndb.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["cardiovascular_disease"], adjacency_tag="GRNDB")
fig, ax = diagnostic.get_diagnostics("components", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_components_cardiovascular_grndb.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["cardiovascular_disease"], adjacency_tag="GRNDB")
fig, ax = diagnostic.get_diagnostics("paths", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_paths_cardiovascular_grndb.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["cardiovascular_disease"], adjacency_tag="GRNDB")
fig, ax = diagnostic.get_diagnostics("degrees", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_degrees_cardiovascular_grndb.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="GRNDB")
fig, ax = diagnostic.get_diagnostics("homophily", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_homophily_immune_grndb.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="IntAct_Direct")
fig, ax = diagnostic.get_diagnostics(save=False)
plt.savefig(os.path.join(config.model.plot_dir, "focus_immune_intact_direct.png"))
"""
"""
#diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="", merge=True)
#fig, ax = diagnostic.get_diagnostics(save=False)
#plt.savefig(os.path.join(config.model.plot_dir, "focus_immune_all_adj.png"))
"""
"""
config.input.randomize_adjacency_percent = 100
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="IntAct_Direct")
fig, ax = diagnostic.get_diagnostics(save=False)
plt.savefig(os.path.join(config.model.plot_dir, "focus_immune_intact_direct_randomized.png"))
"""
"""
config.input.randomize_adjacency_percent = 100
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["cardiovascular_disease"], adjacency_tag="IntAct_Direct")
fig, ax = diagnostic.get_diagnostics(save=False)
plt.savefig(os.path.join(config.model.plot_dir, "focus_cardio_intact_direct_randomized.png"))
"""
"""
config.input.randomize_adjacency_percent = 100
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["cardiovascular_disease"], adjacency_tag="", merge=True)
fig, ax = diagnostic.get_diagnostics(save=False)
plt.savefig(os.path.join(config.model.plot_dir, "focus_cardio_all_randomized.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["cardiovascular_disease"], adjacency_tag="", merge=True)
fig, ax = diagnostic.get_diagnostics(save=False)
plt.savefig(os.path.join(config.model.plot_dir, "focus_cardio_all.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="BioPlex 3.0 293T")
fig, ax = diagnostic.get_diagnostics("metrics", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_metrics_bioplex.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="")
fig, ax = diagnostic.get_diagnostics("metrics", save=True)
plt.savefig(os.path.join(config.model.plot_dir, "panorama_metrics_all_immune.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="IntAct_Direct")
fig, ax = diagnostic.get_diagnostics("metrics", save=True)
plt.savefig(os.path.join(config.model.plot_dir, "metrics_intactdirect_immune.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["cardiovascular_disease"], adjacency_tag="IntAct_Direct")
fig, ax = diagnostic.get_diagnostics("metrics", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "metrics_intactdirect_cardio.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["cardiovascular_disease"], adjacency_tag="BioPlex 3.0 293T")
fig, ax = diagnostic.get_diagnostics("metrics", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "metrics_bioplex_cardio.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="BioPlex 3.0 293T")
fig, ax = diagnostic.get_diagnostics("metrics", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "metrics_bioplex_immune.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="", merge=True)
fig, ax = diagnostic.get_diagnostics("metrics", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "metrics_all_immune.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["cardiovascular_disease"], adjacency_tag="", merge=True)
fig, ax = diagnostic.get_diagnostics("metrics", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "metrics_all_cardio.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["cardiovascular_disease"], adjacency_tag="Recon3D")
fig, ax = diagnostic.get_diagnostics("metrics", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "metrics_recon3d_cardio.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="Recon3D")
fig, ax = diagnostic.get_diagnostics("metrics", save=False)
plt.savefig(os.path.join(config.model.plot_dir, "metrics_recon3d_immune.png"))
"""
"""
diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation", "cardiovascular_disease"], adjacency_tag=["BioPlex 3.0 293T", "HuRI", "IntAct_Direct", "STRING"])
fig, ax = diagnostic.get_diagnostics("homophily", save=False)
plt.tight_layout()
plt.savefig(os.path.join(config.model.plot_dir, "homophily_intact_string.png"))


diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation", "cardiovascular_disease"], adjacency_tag=["BioPlex 3.0 293T", "HuRI", "IntAct_Direct", "STRING"])
fig, ax = diagnostic.get_diagnostics("degrees", save=False)
plt.tight_layout()
plt.savefig(os.path.join(config.model.plot_dir, "degrees_intact_string.png"))

diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation", "cardiovascular_disease"], adjacency_tag=["BioPlex 3.0 293T", "HuRI", "IntAct_Direct", "STRING"])
fig, ax = diagnostic.get_diagnostics("paths", save=False)
plt.tight_layout()
plt.savefig(os.path.join(config.model.plot_dir, "paths_intact_string.png"))

diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation", "cardiovascular_disease"], adjacency_tag=["BioPlex 3.0 293T", "HuRI", "IntAct_Direct", "STRING"])
fig, ax = diagnostic.get_diagnostics("components", save=False)
plt.tight_layout()
plt.savefig(os.path.join(config.model.plot_dir, "components_intact_string.png"))
"""

diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="IntAct_Direct", features=True)
fig, ax = diagnostic.get_diagnostics("paths", save=False, figsize=(8*cm, 8*cm))

plt.tight_layout()
plt.savefig(os.path.join(config.model.plot_dir, "focus_immune_intact_direct_paths.svg"), bbox_inches="tight", dpi=350)

#diagnostic = GraphDiagnosticWrapper(config=config, phenotype_tag=["immune_dysregulation"], adjacency_tag="", merge=True, features=True)
#fig, ax = diagnostic.get_diagnostics(save=False)
#plt.tight_layout()
#plt.savefig(os.path.join(config.model.plot_dir, "focus_immune_all.pdf"), bbox_inches="tight")
