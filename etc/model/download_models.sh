#!/bin/bash

# Get super user privileges
if [[ $EUID != 0 ]]; then
    export wasnt_root=true
	sudo -E "$0" "$@"
fi

if [[ $EUID == 0 ]]; then   
    wget_installed=false
    if command -v wget > /dev/null ; then
        wget_installed=true
    else
        echo "|>>| wget required"
    fi

    if [[ -z `command -v wget` ]] || [[ $wget_installed == false ]]; then
        echo "|>>| installing wget...";
        apt-get update
        apt-get install --no-install-recommends -y wget ca-certificates
    fi
fi

if [[ $EUID != 0 || -z ${wasnt_root} ]]; then
    WGET="wget --retry-connrefused --read-timeout=20 --timeout=15 -t 0 --continue -c"
    FILES="haarcascade_frontalface_default.xml haarcascade_frontalface_alt.xml
    haarcascade_frontalface_alt2.xml haarcascade_frontalface_alt_tree.xml"

    for file in ${FILES}; do
        if [ -e $file ]; then
            echo "|>>| model $file exists, skipping download..."
        else
            url="https://raw.githubusercontent.com/opencv/opencv/4.4.0/data/haarcascades/${file}"
            echo "|>>| downloading ${file} "
            $WGET $url
        fi
    done
fi
