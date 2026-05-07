def compute_poles(comp_A, comp_B, P_bar, xD, xW, zF, R, q, x_arr, y_arr, HL_arr, HV_arr):
    # Entalpias no topo
    HL_xD = interp_HL(xD, x_arr, HL_arr)
    idx_xD = np.argmin(np.abs(x_arr - xD))
    y_top = y_arr[idx_xD]
    HV_top = interp_HV_from_y(y_top, y_arr, HV_arr, x_arr)

    # Polo de retificação (operação normal)
    HD_p = (R + 1) * HV_top - R * HL_xD

    # Alimentação
    HL_zF = interp_HL(zF, x_arr, HL_arr)
    HV_zF = interp_HV_from_y(zF, y_arr, HV_arr, x_arr)
    HF = (1 - q) * HV_zF + q * HL_zF

    # Polo de esgotamento (colinearidade)
    if abs(xD - zF) < 1e-9:
        HW_p = HF
    else:
        slope = (HD_p - HF) / (xD - zF)
        HW_p = HF + slope * (xW - zF)

    # ----- CÁLCULO DO REFLUXO MÍNIMO (CORRIGIDO) -----
    # Para cada tie‑line com xL < xD, calcula H'D e guarda apenas os que são > HV_top
    H_valid = []
    for i in range(len(x_arr)):
        xL = x_arr[i]
        y = y_arr[i]
        HL = HL_arr[i]
        HV = HV_arr[i]
        if abs(y - xL) < 1e-8:
            continue
        if xL >= xD:   # Ignora tie‑lines à direita do destilado
            continue
        # Inclinação da tie‑line
        slope_tl = (HV - HL) / (y - xL)
        H_at_xD = HL + slope_tl * (xD - xL)
        # Só aceita valores fisicamente possíveis (acima da entalpia do vapor de topo)
        if H_at_xD > HV_top + 1e-6:
            H_valid.append(H_at_xD)

    if H_valid:
        H_min = min(H_valid)   # menor H'D entre as tie‑lines válidas
        Rm = (H_min - HV_top) / (HV_top - HL_xD)
        if Rm < 0:
            Rm = None   # segurança
    else:
        Rm = None

    return HD_p, HW_p, HF, Rm, HL_xD, HV_top, y_top