"""Causal graph view widget."""
import math
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem
from PySide6.QtGui import QPen, QBrush, QColor, QFont
from PySide6.QtCore import Qt

class CausalGraphView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene=QGraphicsScene(self)
        self.setScene(self.scene)
        self._positions={}
    def render_graph(self, graph):
        self.scene.clear()
        self._positions={}
        nodes=list(graph.nodes.values())
        n=len(nodes)
        cx,cy=320,260; r=200
        for i,item in enumerate(nodes):
            ang=2*math.pi*i/max(1,n)
            x=cx+r*math.cos(ang); y=cy+r*math.sin(ang)
            self._positions[item.id]=(x,y)
            color="#2ecc71" if item.observed else "#9b59b6"
            el=self.scene.addEllipse(x-22,y-22,44,44,QPen(Qt.black),QBrush(QColor(color)))
            el.setToolTip(item.id + (" (latent)" if item.latent else ""))
            txt=self.scene.addText(item.id,QFont("Arial",9))
            txt.setPos(x-18,y+24)
        for e in graph.edges:
            if e.source in self._positions and e.target in self._positions:
                x1,y1=self._positions[e.source]; x2,y2=self._positions[e.target]
                pen=QPen(QColor("#e74c3c" if e.relation=="confounding" else "#2c3e50"))
                pen.setWidth(2)
                self.scene.addLine(x1,y1,x2,y2,pen)
                # arrowhead
                import math as _m
                ang=_m.atan2(y2-y1,x2-x1)
                ax=x2-12*_m.cos(ang); ay=y2-12*_m.sin(ang)
                self.scene.addLine(ax,ay,x2,y2,pen)
