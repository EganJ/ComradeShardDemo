#!/bin/bash
# Recopy pkgs/ from the dev checkouts' working trees and print the commit each copy comes
# from, with any uncommitted paths it includes. ReactantCore comes from the Reactant tree
# pinned in Manifest.toml.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEV="${JULIA_DEV:-$HOME/.julia/dev}"
SOURCES=(
    Comrade:Comrade-Redesign
    ComradeBase:ComradeBase-Shard
    VLBISkyModels:VLBISkyModels-Shard
    VLBIInstrumentModels:VLBIInstrumentModels
    VLBIImagePriors:VLBIImagePriors
    VLBILikelihoods:VLBILikelihoods
    VLBIData:VLBIData
    InterferometricModels:InterferometricModels
    FieldDimArrays:FieldDimArrays
)
PARTS=(src ext Project.toml LICENSE README.md)

copy() {  # copy <source dir> <pkgs/ name>
    local src=$1 dst=$HERE/pkgs/$2
    [ -f "$src/Project.toml" ] || { echo "no package at $src" >&2; exit 1; }
    rm -rf "$dst"
    mkdir -p "$dst"
    for p in "${PARTS[@]}"; do
        [ -e "$src/$p" ] && cp -r "$src/$p" "$dst/"
    done
    return 0
}

for s in "${SOURCES[@]}"; do
    name=${s%%:*}
    src=$DEV/${s#*:}
    copy "$src" "$name"
    present=()
    for p in "${PARTS[@]}"; do [ -e "$src/$p" ] && present+=("$p"); done
    dirty=$(git -C "$src" status --short -- "${present[@]}" | awk '{print $2}' | tr '\n' ' ')
    printf '%-22s %-18s %s%s\n' "$name" "$(git -C "$src" rev-parse --abbrev-ref HEAD)" \
        "$(git -C "$src" rev-parse --short HEAD)" "${dirty:+  + uncommitted: $dirty}"
done

rdir=$(julia --startup-file=no -e '
    using TOML
    r = only(TOML.parsefile(ARGS[1])["deps"]["Reactant"])
    slug = Base.version_slug(Base.UUID(r["uuid"]), Base.SHA1(r["git-tree-sha1"]))
    print(joinpath(DEPOT_PATH[1], "packages", "Reactant", slug))' "$HERE/Manifest.toml")
[ -d "$rdir/lib/ReactantCore" ] || { echo "Reactant tree not installed: $rdir; run Pkg.instantiate()" >&2; exit 1; }
copy "$rdir/lib/ReactantCore" ReactantCore
printf '%-22s %s\n' ReactantCore "lib/ReactantCore of Reactant at $rdir"
