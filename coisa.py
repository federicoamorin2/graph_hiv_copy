def make_fluxo_resultados(path="fig.svg"):
    config = Config(SCALE=1.3)
    calculadora = CalculadoraBeneficio()
    labels = BeneficioStrings(calculadora)
    builder = Builder(config)
    
    # Base
    base = builder.add_decision("Estrategista")
    
    # Estrategista == Defesa
    defesa = builder.add_decision('Modelo', pos=(10, 8))
    builder.add_arrow(at=base.N, to=defesa.W)
    
    defesa_acordo = builder.add_decision('Resultados', pos=(15, 4), north='Condenacao', south="Sem Onus")
    builder.add_arrow(at=defesa.S, to=defesa_acordo.W)
    
    defesa_acerto = builder.add_circle(label=labels.defesa_acerto, pos=(20, 6), color=Colors.GREEN)
    builder.add_arrow(at=defesa_acordo.N, to=defesa_acerto.W)

    defesa_erro = builder.add_circle(label=labels.defesa_erro, pos=(20, 2), color=Colors.RED)
    builder.add_arrow(at=defesa_acordo.S, to=defesa_erro.W)

    defesa_sem_impacto = builder.add_circle(label=labels.defesa_sem_impacto, pos=(15, 12), color=Colors.GREY)
    builder.add_arrow(at=defesa.N, to=defesa_sem_impacto.W)

    # Estrategista == Acordo
    acordo = builder.add_decision('Modelo', pos=(10, -8))
    builder.add_arrow(at=base.S, to=acordo.W)
    
    acordo_defesa = builder.add_decision('Resultados', pos=(15, -4), north='Condenacao', south="Sem Onus")
    builder.add_arrow(at=acordo.N, to=acordo_defesa.W)
    
    acordo_erro = builder.add_circle(label=labels.acordo_erro, pos=(20, -2), color=Colors.RED)
    builder.add_arrow(at=acordo_defesa.N, to=acordo_erro.W)

    acordo_acerto = builder.add_circle(label=labels.acordo_acerto, pos=(20, -6), color=Colors.GREEN)
    builder.add_arrow(at=acordo_defesa.S, to=acordo_acerto.W)

    acordo_sem_impacto = builder.add_circle(label=labels.acordo_sem_impacto, pos=(15, -12), color=Colors.GREY)
    builder.add_arrow(at=acordo.S, to=acordo_sem_impacto.W)
    builder.save(path)
    return builder
