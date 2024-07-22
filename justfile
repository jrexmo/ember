default:
  just --list

add:
  git add -A

commit msg="updates": add
  git commit -m "{{msg}}"

push: add commit
  git push
  
