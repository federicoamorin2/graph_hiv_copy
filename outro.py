from schemdraw import flow
import schemdraw
import schemdraw.elements as elm
from enum import Enum
from dataclasses import dataclass

class Colors(str, Enum):
    GREY = "#999999"
    RED = "#f44336"
    BLACK = "#000000"
    GREEN = "#acf30c"

@dataclass
class Config:
    CIRCLE_SIZE: float = 1.4
    SCALE: int = 1

    @property
    def scaled_circle(self):
        return self.CIRCLE_SIZE * self.SCALE

class BeneficioStrings:
    def __init__(self, calculadora):
        self.calculadora = calculadora

    @property
    def acordo_sem_impacto(self):
        return f"Sem impacto:\n{self.calculadora.vol_acordo_sem_impacto}"

    @property
    def defesa_sem_impacto(self):
        return f"Sem impacto:\n{self.calculadora.vol_defesa_sem_impacto}"

    @property
    def defesa_acerto(self):
        return f'Acertos\nVol: {self.calculadora.vol_acertos_defesa}\nBeneficio unitario:\nR$ {self.calculadora.vol_acertos_defesa}\nBeneficio total:\nR$ {self.calculadora.beneficio_defesa:.2f}'

    @property
    def defesa_erro(self):
        return f'Erros\nVol: {self.calculadora.vol_erros_defesa}\nMaleficio unitario:\nR$ {self.calculadora.maleficio_unitario_defesa}\nMaleficio total:\nR${self.calculadora.maleficio_defesa}'

    @property
    def acordo_acerto(self):
        return f'Acertos\nVol: {self.calculadora.vol_acertos_acordo}\nBeneficio unitario:\nR$ {self.calculadora.beneficio_unitario_acordo:.2f}\nBeneficio total:\nR$ {self.calculadora.beneficio_acordo:.2f}'

    @property
    def acordo_erro(self):
        return f'Erros\nVol: {self.calculadora.vol_erros_acordo}\nMaleficio unitario:\n{self.calculadora.maleficio_unitario_acordo:.2f}\nMaleficio total:\n{self.calculadora.maleficio_acordo:.2f}'

    @property
    def beneficio_total(self):
        return f"Beneficio Total:\n{self.calculadora.beneficio}\nBeneficio Acordo:\n{self.calculadora.beneficio_total_acordo:.2f}\nBeneficio Defesa:\n{self.calculadora.beneficio_total_defesa:.2f}"


class Builder:
    def __init__(self, config, unit=1, backend='svg'):
        self.config = config
        self.drawing = schemdraw.Drawing(unit=unit, backend=backend)
        
    def add_element(self, element):
        self.drawing += element

    def save(self, name):
        self.drawing.save(name)

    def move(self, pos):
        self.drawing.here = pos
    
    def add_arrow(self, at, to):
        arrow = elm.OrthoLines(arrow='->').at(at).to(to)
        self.add_element(arrow)
        return arrow

    def add_decision(self, label, pos=(0, 0), north="Defesa", south="Acordo"):
        element = flow.Decision(
            w=3*self.config.SCALE,
            h=3*self.config.SCALE,
            N=north,
            S=south
        ).label(label)
        self.move(pos)
        self.add_element(element)
        return element

    def add_circle(self, label, pos, color=Colors.GREY, fill=True):
        circle = flow.Circle(r=self.config.scaled_circle).label(label)
        self.move(pos)
        self.add_element(circle)
        circle.color(color.value)
        circle.fill(fill)
        return circle
