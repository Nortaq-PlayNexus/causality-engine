"""Main window."""
import sys, random
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QSplitter, QGroupBox, QLineEdit)
from PySide6.QtCore import Qt
from .causal_graph_view import CausalGraphView
from ..core import (Ecosystem, PopulationDynamics, ResourceSystem, PhysicalModel, ArtificialEconomy,
    CausalGraph, Node, Edge, SCM, HypothesisSet, pc_skeleton, score_based_hill_climbing,
    shd, false_discovery_rate, true_positive_rate, pearson, spurious_correlation_warning)
from ..core.storage import Storage
from ..benchmark.runner import run_all

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAUSALITY ENGINE")
        self.resize(1200,800)
        self.graph=None
        self.scm=None
        self.data=None
        self.system=None
        self.storage=Storage("causality.db")
        self._build()

    def _build(self):
        tabs=QTabWidget()
        self.setCentralWidget(tabs)
        # Tab 1: Simulation
        sim=QWidget(); sl=QVBoxLayout(sim)
        top=QHBoxLayout()
        self.sys_combo=QComboBox(); self.sys_combo.addItems(["Ecosystem","PopulationDynamics","ResourceSystem","PhysicalModel","ArtificialEconomy"])
        top.addWidget(QLabel("System:")); top.addWidget(self.sys_combo)
        self.seed_spin=QSpinBox(); self.seed_spin.setRange(0,9999); self.seed_spin.setValue(42)
        top.addWidget(QLabel("Seed:")); top.addWidget(self.seed_spin)
        self.T_spin=QSpinBox(); self.T_spin.setRange(10,500); self.T_spin.setValue(60)
        top.addWidget(QLabel("T:")); top.addWidget(self.T_spin)
        btn=QPushButton("Build & Run"); btn.clicked.connect(self.run_sim)
        top.addWidget(btn)
        sl.addLayout(top)
        self.graph_view=CausalGraphView()
        sl.addWidget(self.graph_view)
        self.sim_table=QTableWidget()
        sl.addWidget(self.sim_table)
        tabs.addTab(sim,"Simulation")
        # Tab 2: Interventions
        iv=QWidget(); ivl=QVBoxLayout(iv)
        ivl.addWidget(QLabel("Intervention: target node"))
        self.iv_target=QLineEdit(); ivl.addWidget(self.iv_target)
        self.iv_value=QDoubleSpinBox(); self.iv_value.setRange(-100,100); self.iv_value.setValue(5.0)
        ivl.addWidget(self.iv_value)
        biv=QPushButton("Apply do() intervention"); biv.clicked.connect(self.apply_intervention)
        ivl.addWidget(biv)
        self.iv_out=QTextEdit(); self.iv_out.setReadOnly(True)
        ivl.addWidget(self.iv_out)
        tabs.addTab(iv,"Interventions")
        # Tab 3: Counterfactuals
        cf=QWidget(); cfl=QVBoxLayout(cf)
        cfl.addWidget(QLabel("Counterfactual: observed JSON (node:value)"))
        self.cf_obs=QTextEdit(); self.cf_obs.setPlainText('{"X0": 1.0}')
        cfl.addWidget(self.cf_obs)
        cfl.addWidget(QLabel("Intervention JSON"))
        self.cf_iv=QTextEdit(); self.cf_iv.setPlainText('{"X1": 0.0}')
        cfl.addWidget(self.cf_iv)
        bcf=QPushButton("Compute counterfactual"); bcf.clicked.connect(self.compute_cf)
        cfl.addWidget(bcf)
        self.cf_out=QTextEdit(); self.cf_out.setReadOnly(True)
        cfl.addWidget(self.cf_out)
        tabs.addTab(cf,"Counterfactuals")
        # Tab 4: Graph Recovery
        gr=QWidget(); grl=QVBoxLayout(gr)
        bgr=QPushButton("Run PC skeleton"); bgr.clicked.connect(self.run_pc)
        grl.addWidget(bgr)
        bhc=QPushButton("Run Hill-Climbing"); bhc.clicked.connect(self.run_hc)
        grl.addWidget(bhc)
        self.gr_metrics=QTextEdit(); self.gr_metrics.setReadOnly(True)
        grl.addWidget(self.gr_metrics)
        tabs.addTab(gr,"Graph Recovery")
        # Tab 5: Hypotheses
        hy=QWidget(); hyl=QVBoxLayout(hy)
        bhy=QPushButton("Compare alternative hypotheses"); bhy.clicked.connect(self.compare_hyps)
        hyl.addWidget(bhy)
        self.hy_out=QTextEdit(); self.hy_out.setReadOnly(True)
        hyl.addWidget(self.hy_out)
        tabs.addTab(hy,"Hypotheses")
        # Tab 6: Benchmark
        bm=QWidget(); bml=QVBoxLayout(bm)
        bbm=QPushButton("Run Full Benchmark"); bbm.clicked.connect(self.run_benchmark)
        bml.addWidget(bbm)
        self.bm_out=QTextEdit(); self.bm_out.setReadOnly(True)
        bml.addWidget(self.bm_out)
        tabs.addTab(bm,"Benchmark")
        # Tab 7: Correlation caveat
        ca=QWidget(); cal=QVBoxLayout(ca)
        cal.addWidget(QLabel("Correlation != Causation. Enter two node series? Use simulation data."))
        bca=QPushButton("Show correlation vs causation report"); bca.clicked.connect(self.corr_report)
        cal.addWidget(bca)
        self.ca_out=QTextEdit(); self.ca_out.setReadOnly(True)
        cal.addWidget(self.ca_out)
        tabs.addTab(ca,"Correlation")

    def _make_system(self):
        seed=self.seed_spin.value()
        name=self.sys_combo.currentText()
        if name=="Ecosystem": self.system=Ecosystem(seed=seed)
        elif name=="PopulationDynamics": self.system=PopulationDynamics(seed=seed)
        elif name=="ResourceSystem": self.system=ResourceSystem(seed=seed)
        elif name=="PhysicalModel": self.system=PhysicalModel(seed=seed)
        elif name=="ArtificialEconomy": self.system=ArtificialEconomy(seed=seed)
        self.graph=self.system.graph

    def run_sim(self):
        self._make_system()
        T=self.T_spin.value()
        self.data=self.system.run(T=T)
        self.graph_view.render_graph(self.graph)
        # show last 10 rows
        if self.data:
            keys=list(self.data[0].keys())
            self.sim_table.setColumnCount(len(keys))
            self.sim_table.setHorizontalHeaderLabels(keys)
            self.sim_table.setRowCount(min(10,len(self.data)))
            for i,row in enumerate(self.data[-10:]):
                for j,k in enumerate(keys):
                    self.sim_table.setItem(i,j,QTableWidgetItem(f"{row[k]:.3f}"))
        # build scm
        self.scm=SCM(self.graph, noise_std=0.1, seed=self.seed_spin.value())
        QMessageBox.information(self,"Done","Simulation complete. Ground-truth causal graph shown.")

    def apply_intervention(self):
        if self.scm is None:
            QMessageBox.warning(self,"","Run simulation first."); return
        tgt=self.iv_target.text().strip()
        val=self.iv_value.value()
        iv={tgt:val}
        samples=self.scm.do(iv, n_samples=5, seed=1)
        out=f"do({tgt}={val}):\n"
        for s in samples:
            out+="  "+str({k:round(v,3) for k,v in s.items()})+"\n"
        self.iv_out.setPlainText(out)

    def compute_cf(self):
        if self.scm is None:
            QMessageBox.warning(self,"","Run simulation first."); return
        import json
        try:
            obs=json.loads(self.cf_obs.toPlainText())
            iv=json.loads(self.cf_iv.toPlainText())
        except Exception as e:
            QMessageBox.warning(self,"",str(e)); return
        res=self.scm.counterfactual(obs, iv, seed=1)
        self.cf_out.setPlainText(str({k:round(v,3) for k,v in res.items()}))

    def run_pc(self):
        if self.data is None:
            QMessageBox.warning(self,"","Run simulation first."); return
        keys=list(self.data[0].keys())
        rows=[{k:d[k] for k in keys} for d in self.data]
        g=pc_skeleton(rows, keys, alpha=0.15)
        self.graph_view.render_graph(g)
        m=f"PC skeleton edges: {len(g.edges)}\nTrue edges: {len(self.graph.edges)}\n"
        m+=f"SHD: {shd(g,self.graph)}\nFDR: {false_discovery_rate(g,self.graph):.3f}\nTPR: {true_positive_rate(g,self.graph):.3f}\n"
        self.gr_metrics.setPlainText(m)

    def run_hc(self):
        if self.data is None:
            QMessageBox.warning(self,"","Run simulation first."); return
        keys=list(self.data[0].keys())
        rows=[{k:d[k] for k in keys} for d in self.data]
        g=score_based_hill_climbing(rows, keys)
        self.graph_view.render_graph(g)
        m=f"Hill-climbing edges: {len(g.edges)}\nTrue edges: {len(self.graph.edges)}\n"
        m+=f"SHD: {shd(g,self.graph)}\nFDR: {false_discovery_rate(g,self.graph):.3f}\nTPR: {true_positive_rate(g,self.graph):.3f}\n"
        self.gr_metrics.setPlainText(m)

    def compare_hyps(self):
        if self.graph is None:
            QMessageBox.warning(self,"","Run simulation first."); return
        hs=HypothesisSet()
        hs.add(self.graph, "ground_truth", prior=0.5)
        # alternative: reversed edges
        g2=CausalGraph()
        for n in self.graph.nodes.values():
            g2.add_node(Node(id=n.id, observed=n.observed, latent=n.latent))
        for e in self.graph.edges:
            g2.add_edge(Edge(e.target, e.source, e.weight, e.relation))
        hs.add(g2, "reversed", prior=0.25)
        # empty
        g3=CausalGraph()
        for n in self.graph.nodes.values():
            g3.add_node(Node(id=n.id, observed=n.observed, latent=n.latent))
        hs.add(g3, "empty", prior=0.25)
        def score_fn(g):
            # lower shd to true = higher score
            return -shd(g,self.graph)
        hs.update_scores(score_fn)
        out=""
        for h in hs.hypotheses:
            out+=f"{h['name']}: posterior={h['posterior']:.3f} score={h['score']:.3f}\n"
        out+="\nBest: "+(hs.best()['name'] if hs.best() else "none")
        self.hy_out.setPlainText(out)

    def run_benchmark(self):
        res=run_all()
        import json
        self.bm_out.setPlainText(json.dumps(res, indent=2))

    def corr_report(self):
        if self.data is None:
            QMessageBox.warning(self,"","Run simulation first."); return
        keys=list(self.data[0].keys())
        out="Correlation matrix (Pearson):\n"
        for a in keys:
            for b in keys:
                if a<b:
                    xa=[d[a] for d in self.data]; xb=[d[b] for d in self.data]
                    r=pearson(xa,xb)
                    out+=f"  {a}~{b}: r={r:.3f}  {spurious_correlation_warning(r)}\n"
        out+="\nCAUSATION REQUIRES: "+", ".join(__import__("causality_engine.core.correlation",fromlist=["CAUSATION_REQUIRES"]).CAUSATION_REQUIRES)
        self.ca_out.setPlainText(out)

def main():
    app=__import__("PySide6.QtWidgets",fromlist=["QApplication"]).QApplication(sys.argv)
    w=MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()
