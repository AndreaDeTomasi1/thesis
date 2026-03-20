#let titlefont = "Libertinus Serif"
#let textfont = "Quattrocento Sans"
#let citationfont = "STIX Two Text"

#let authorsize = 14pt
#let datesize = 12pt

#let titlesize = 32pt
#let header1size = 28pt
#let header2size = 18pt
#let header3size = 14pt

#let textsize = 12pt
#let footnotesize = 10pt

// Title page
#let titlepage(title: "", subtitle: "", authors: "", date: "", logo: none) = {
  if title != "" or authors != "" {
    set page(header: none, footer: none)

    // Logo in top right if provided
    if logo != none {
      place(top + right, dx: 0pt, dy: 0pt, {
        image(logo, width: 5cm)
      })
    }

    // Center the title content vertically
    v(1fr)
    align(center, {
      std.title(par(leading: 1em, title))
      v(16pt)
      if subtitle != "" {
        par(
          leading: 1.1em,
          text(size: header2size, weight: "bold", font: titlefont, style: "italic", subtitle)
        )
        v(16pt)
      }
      if authors != "" {
        text(size: authorsize, authors)
        v(8pt)
      }
      if date != "" {
        text(size: datesize, date)
      }
    })
    v(1fr)

    pagebreak()
  }
}

#let ottante-report(
  title: "",
  subtitle: "",
  authors: "",
  date: "",
  logo: none,
  left-header-content: "",
  right-header-content: "",
  unnumbered-sections: false,
  body
) = {

  // Set document metadata
  set document(title: title, author: authors)

  // Page setup and margins
  set page(
    paper: "a4",
    margin: (
      top: auto,
      bottom: auto,
      inside: auto,
      outside: auto,
    ),
    header: {
      set text(fill: rgb("#666666"))
      v(0.3em)
      grid(
        columns: (1fr, 1fr, 1fr),
        [#left-header-content],
        [],
        align(right, [#right-header-content]),
      )
      v(0.3em)
      line(length: 100%, stroke: 0.5pt)
    },
    footer: context {
      set text(fill: rgb("#666666"))
      grid(
        columns: (1fr, 1fr, 1fr),
        [],
        [],
        align(right, [#counter(page).display()]),
      )
    }
  )

  // Configure paragraph spacing and indentation
  set par(
    leading: 1em,
    spacing: 1.5em,
    first-line-indent: 0pt,
  )

  // Set text justification to left-aligned (ragged right margin)
  set par(justify: false)

  // Define fonts
  set text(
    font: textfont,
    size: 12pt,
    lang: "en",
  )

  // Configure title and headings
  show title: set text(weight: "bold", font: titlefont, size: titlesize)

  set heading(numbering: if unnumbered-sections { none } else { "1.1.1" })
  show heading: set text(weight: "bold", font: titlefont)

  show heading.where(level: 1): set text(size: header1size)
  show heading.where(level: 1): set block(above: 1.8em, below: 1.5em)

  show heading.where(level: 2): set text(size: header2size)
  show heading.where(level: 2): set block(above: 1.44em, below: 1.2em)

  show heading.where(level: 3): set text(size: header3size)
  show heading.where(level: 3): set block(above: 1.2em, below: 1em)

  // Configure lists
  set list(spacing: 1em)
  show list.item: set par(leading: 0.6em)

  set enum(spacing: 1em)
  show enum.item: set par(leading: 0.6em)

  // Configure tables
  set table(
    align: left,
    // none can be overridden, but 0pt can't
    stroke: (_, y) => (
        left: { 0pt },
        right: { 0pt },
        top: if y == 0 { 1pt } else if y == 1 { none } else { 0pt },
        bottom: if y == 0 { 0.5pt } else { 1pt }
    )
  )
  show table.cell.where(y: 0): set text(weight: "bold")

  // Configure quotes
  show quote: set text(style: "italic", font: citationfont)

  // Configure figure captions
  show figure.caption: set text(style: "italic")

  show figure: it => {
    block(
      above: 2em,
      below: 2em,
      {
        box(
          width: 100%,
          it.body,
        )
        it.caption
      }
    )
  }

  // Configure links and references
  set cite(style: "alphanumeric")
  show link: underline

  // Configure footnotes
  set footnote.entry(
    separator: line(length: 100%, stroke: 0.5pt) + v(6pt),
  )
  show footnote.entry: set text(size: footnotesize)

  // Title page

  titlepage(title: title, subtitle: subtitle, authors: authors, date: date, logo: logo)

  // Render document body
  body

}

#let smallsection(body) = {
  heading(level: 3, numbering: none, outlined: false, body)
}

#let disclaimer(recipient) = {
  text(size: footnotesize)[The information contained in this report is proprietary and confidential. It is intended solely for the use of ]
  text(size: footnotesize, recipient)
  text(size: footnotesize)[ and may not be disclosed to or used by any other subject without prior written consent from Ottante.
  This report represents the current state of knowledge and research conducted by Ottante at the time of preparation,
  but it does not constitute a warranty, guarantee, or representation that the results or advances described herein will be achieved in future projects.
  The information contained in this report is based on our best efforts and sources believed to be reliable, but it may contain errors, omissions, or inaccuracies.
  Ottante does not accept any liability for any loss or damage caused by the use of this report, whether directly or indirectly.]
}

#let copyright(year) = {
  text(size: textsize)[© ]
  text(size: textsize, year)
  text(size: textsize)[ Ottante.]
}

#let address() = {
  text(size: footnotesize)[
  Via Socrate, 41\
  20128, Milano\
  Italy
  ]
}

#let contacts(values) = {
  text(size: footnotesize)[
    #for value in values [
      #value\
    ]
  ]
}

// Table of Contents
#let tableofcontents() = {
  outline(
    title: [Table of Contents],
    indent: auto,
  )
}

// Remark environment
#let remark(content) = {
  block(
    fill: rgb("#f0f0f0"),
    inset: 12pt,
    radius: 4pt,
    {
      v(0.5em)
      text(weight: "bold", "Remark: ")
      text(content)
      v(0.5em)
    }
  )
}
