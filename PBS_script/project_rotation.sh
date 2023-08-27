#!/bin/bash

get_project () {

    project_ls=("phys_DL" "PDLAI" "dnn_maths" "ddl" "dyson" "vortex_dl" "frac_attn")
    MAIN_DIR="/project/frac_attn"
    cd $MAIN_DIR
    DIR="project_tracker"
    if [ ! -d "${DIR}" ]; then
        mkdir "${DIR}"
    fi

    cd "project_tracker"
    if [ ! -f "previous_project.txt" ]; then
        current_project="${project_ls[0]}"
        echo "${current_project}" > previous_project.txt
        echo "There is no previous project, now set to ${project_ls[0]}"
    else
        previous_project=`cat previous_project.txt | head -n1 | awk '{print $1;}'`
        for i in "${!project_ls[@]}"; do
            if [[ "${project_ls[$i]}" = "${previous_project}" ]]; then
                if [ ! $i -eq $((${#project_ls[@]} - 1)) ]; then
                    current_project="${project_ls[$(($i + 1))]}"
                    echo "${current_project}" >| previous_project.txt
                    #echo "The previous project was ${project_ls[$i]}, current project: ${current_project}"   
                else
                    current_project="${project_ls[0]}"
                    echo "${current_project}" >| previous_project.txt
                    #echo "Previous project: ${project_ls[${#project_ls[@]}-1]}, current project: ${project_ls[0]}"
                fi
            fi
        done      
    fi

    echo $current_project
    #return $current_project

}