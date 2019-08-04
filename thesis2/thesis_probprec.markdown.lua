local ran_ok, error = pcall(function() local kpse = require('kpse') kpse.set_program_name('luatex') local lfs = require("lfs") local cacheDir = "./_markdown_thesis_probprec" if lfs.isdir(cacheDir) == true then else assert(lfs.mkdir(cacheDir)) end local md = require("markdown") local convert = md.new({cacheDir = "./_markdown_thesis_probprec", citations = true, citationNbsps = true, definitionLists = true, footnotes = true, fencedCode = true, hashEnumerators = true, hybrid = true, inlineFootnotes = true, outputDir = ".", } ) local input = assert(io.open("thesis_probprec.markdown.in","r")):read("*a") print(convert(input:gsub("\r\n?", "\n"))) end) if not ran_ok then local file = io.open("./thesis_probprec.markdown.err", "w") if file then file:write(error .. "\n") file:close() end print('\\markdownError{An error was encountered while executing Lua code}{For further clues, examine the file "./thesis_probprec.markdown.err"}') end