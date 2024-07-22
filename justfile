default:
  just --list

add:
  git add -A

commit msg=`git status --short`: add
  git commit -m "{{msg}}"

push: add commit
  git push

feature request:
  rye run python src/ember/llm/main.py add-feature "{{request}}"
