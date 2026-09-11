#!/bin/bash
# Wrapper so the .bat only has to pass one plain argument to bash, instead of
# nesting quoted cd/redirection logic inside a cmd.exe-passed string (fragile
# across invocation contexts -- broke when double-clicked vs run from a shell).
cd "$(dirname "$0")" || exit 1
MODEL="$1" ./pick_hot_layers.sh
