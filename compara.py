import argparse
import csv
import glob
from collections import Counter, defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compara arquivos submission*.csv pela coluna index/answer."
    )
    parser.add_argument(
        "arquivos",
        nargs="*",
        help="Arquivos CSV para comparar. Padrao: submission*.csv.",
    )
    parser.add_argument(
        "-r",
        "--reference",
        help="Arquivo usado como referencia. Padrao: primeiro arquivo em ordem alfabetica.",
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=30,
        help="Quantidade maxima de divergencias detalhadas na tela. Padrao: 30.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Salva todas as divergencias em um CSV.",
    )
    return parser.parse_args()


def carregar_submission(caminho):
    respostas = {}

    with open(caminho, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        colunas = set(reader.fieldnames or [])
        if not {"index", "answer"}.issubset(colunas):
            raise ValueError(f"{caminho} precisa ter as colunas index e answer.")

        for linha in reader:
            respostas[str(linha["index"])] = str(linha["answer"]).strip()

    return respostas


def descobrir_arquivos(arquivos):
    if arquivos:
        return sorted(arquivos)

    return sorted(glob.glob("submission*.csv"))


def valor(respostas_por_arquivo, arquivo, idx):
    return respostas_por_arquivo.get(arquivo, {}).get(idx, "<ausente>")


def comparar(arquivos, referencia, limite):
    respostas_por_arquivo = {
        arquivo: carregar_submission(arquivo)
        for arquivo in arquivos
    }

    indices = sorted(
        {idx for respostas in respostas_por_arquivo.values() for idx in respostas},
        key=lambda idx: int(idx) if idx.isdigit() else idx,
    )
    divergencias = []

    for idx in indices:
        respostas = {
            arquivo: valor(respostas_por_arquivo, arquivo, idx)
            for arquivo in arquivos
        }
        if len(set(respostas.values())) > 1:
            divergencias.append((idx, respostas))

    print(f"Arquivos comparados: {len(arquivos)}")
    for arquivo in arquivos:
        print(f"- {arquivo}: {len(respostas_por_arquivo[arquivo])} linhas")

    print(f"\nTotal de indices comparados: {len(indices)}")
    print(f"Total de divergencias: {len(divergencias)}")

    if referencia:
        print(f"\nReferencia: {referencia}")
        for arquivo in arquivos:
            if arquivo == referencia:
                continue

            iguais = 0
            diferentes = 0
            ausentes = 0

            for idx in indices:
                resposta_ref = valor(respostas_por_arquivo, referencia, idx)
                resposta_arquivo = valor(respostas_por_arquivo, arquivo, idx)
                if resposta_ref == "<ausente>" or resposta_arquivo == "<ausente>":
                    ausentes += 1
                elif resposta_ref == resposta_arquivo:
                    iguais += 1
                else:
                    diferentes += 1

            print(
                f"- {arquivo}: {iguais} iguais, {diferentes} diferentes, "
                f"{ausentes} com indice ausente"
            )

    contagem_respostas = defaultdict(Counter)
    for _, respostas in divergencias:
        for arquivo, resposta in respostas.items():
            contagem_respostas[arquivo][resposta] += 1

    if divergencias:
        print(f"\nPrimeiras divergencias (limite {limite}):")
        cabecalho = ["index"] + arquivos
        print(",".join(cabecalho))
        for idx, respostas in divergencias[:limite]:
            linha = [idx] + [respostas[arquivo] for arquivo in arquivos]
            print(",".join(linha))

        print("\nRespostas em divergencias por arquivo:")
        for arquivo in arquivos:
            resumo = ", ".join(
                f"{resposta}={quantidade}"
                for resposta, quantidade in contagem_respostas[arquivo].most_common()
            )
            print(f"- {arquivo}: {resumo}")

    return divergencias


def salvar_divergencias(caminho_saida, arquivos, divergencias):
    with open(caminho_saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index"] + arquivos)

        for idx, respostas in divergencias:
            writer.writerow([idx] + [respostas[arquivo] for arquivo in arquivos])


def main():
    args = parse_args()
    arquivos = descobrir_arquivos(args.arquivos)

    if len(arquivos) < 2:
        raise SystemExit("Informe pelo menos dois arquivos ou tenha dois submission*.csv na pasta.")

    referencia = args.reference or arquivos[0]
    if referencia not in arquivos:
        arquivos = sorted([referencia] + arquivos)

    divergencias = comparar(arquivos, referencia, args.limit)

    if args.output:
        salvar_divergencias(args.output, arquivos, divergencias)
        print(f"\nDivergencias salvas em: {args.output}")


if __name__ == "__main__":
    main()
