"""CLI del sistema PapaScan Multiagente.

Ejemplos:
    python -m multiagente.run --image ruta/a/hoja.jpg
    python -m multiagente.run --image hoja.jpg --no-llm          # sin LLM (offline puro)
    python -m multiagente.run --image hoja.jpg --json            # salida JSON
    python -m multiagente.run --image hoja.jpg --ask "¿es contagioso?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multiagente.coordinator import Coordinator


def _print_transcript(bb) -> None:
    print("\n" + "=" * 70)
    print("DIÁLOGO ENTRE AGENTES")
    print("=" * 70)
    print(bb.transcript())


def _print_report(rep: dict) -> None:
    print("\n" + "=" * 70)
    print("REPORTE FINAL")
    print("=" * 70)
    print(f"  Caso:            {rep['case_id']}")
    print(f"  Diagnóstico:     {rep['diagnostico']}  "
          f"(confianza {(_pct(rep['confianza']))}, estado {rep['estado_confianza']})")
    print(f"  Modelo:          {rep['version_modelo']}")
    sev = rep.get("severidad", {})
    if sev:
        print(f"  Severidad:       {sev.get('nivel', 'n/d')}")
    print(f"  Atención modelo: {rep['zona_gradcam']}")
    rec = rep.get("recomendacion", {})
    if rec:
        print(f"  Enfermedad (KB): {rec.get('enfermedad')}  | urgencia: {rec.get('urgencia')}")
    if rep["derivar_a_tecnico"]:
        motivos = ", ".join(rep["validacion_confianza"].get("motivos", []))
        print(f"  ⚠ DERIVAR A TÉCNICO: {motivos}")
    print(f"  Explicación por LLM: {rep['explicacion_por_llm']}")

    explic = rep.get("explicacion", {})
    if explic:
        print("\n  --- Explicación al agricultor ---")
        for k in ("resumen", "que_observo_el_modelo", "nivel_confianza",
                  "que_hacer", "alerta", "siguiente_paso"):
            if explic.get(k):
                print(f"  • {k}: {explic[k]}")

    ve = rep.get("validacion_explicacion", {})
    if ve:
        estado = "segura ✓" if ve.get("segura") else "corregida ⚠"
        print(f"\n  Verificación de seguridad: {estado}")

    print("\n  Archivos generados:")
    for k, v in rep.get("archivos", {}).items():
        print(f"    {k}: {v}")


def _pct(x) -> str:
    try:
        return f"{x*100:.0f} %"
    except Exception:
        return str(x)


def main(argv: list[str] | None = None) -> int:
    # La consola de Windows usa cp1252 por defecto y no puede imprimir los
    # acentos/flechas del diálogo. Forzamos UTF-8 en la salida si se puede.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description="PapaScan Multiagente — diagnóstico de papa.")
    parser.add_argument("--image", required=True, help="Ruta a la imagen de la hoja.")
    parser.add_argument("--case-id", default=None, help="Identificador del caso.")
    parser.add_argument("--no-llm", action="store_true", help="No usar el LLM (respaldo determinístico).")
    parser.add_argument("--json", action="store_true", help="Imprimir el reporte como JSON.")
    parser.add_argument("--no-transcript", action="store_true", help="Ocultar el diálogo entre agentes.")
    parser.add_argument("--ask", default=None, help="Pregunta de seguimiento al AgenteConversacional.")
    args = parser.parse_args(argv)

    img = Path(args.image)
    if not img.exists():
        print(f"ERROR: no existe la imagen: {img}", file=sys.stderr)
        return 2
    case_id = args.case_id or img.stem.replace(" ", "_")

    coord = Coordinator(usar_llm=not args.no_llm)
    bb = coord.diagnosticar(str(img), case_id=case_id)
    rep = coord.reporte(bb)

    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        if not args.no_transcript:
            _print_transcript(bb)
        _print_report(rep)

    if args.ask:
        respuesta, _ = coord.conversacional.preguntar(bb, args.ask)
        print("\n" + "=" * 70)
        print("CHAT DE SEGUIMIENTO")
        print("=" * 70)
        print(f"  Pregunta:  {args.ask}")
        print(f"  Respuesta: {respuesta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
