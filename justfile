default:
  just --list

add:
  git add -A

commit msg=`git status --short`: add
  git commit -m "{{msg}}"

push: add commit
  git push

