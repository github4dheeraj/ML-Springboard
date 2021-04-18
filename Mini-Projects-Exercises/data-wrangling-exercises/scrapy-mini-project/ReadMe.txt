===================
Scrapy Mini Project
===================

This mini-project is simply a tutorial on how to build scrapy spiders, and is
for most part a copy of the original excellent tutorial available at: `<https://docs.scrapy.org/en/latest/intro/tutorial.html>`_

In this tutorial, we'll assume that Scrapy is already installed on your system.
If that's not the case, please simply run `pip install scrapy`_.

We are going to scrape `quotes.toscrape.com <http://quotes.toscrape.com/>`_, an
synthetic website that lists quotes from famous authors.

This tutorial will walk you through these tasks:

1.Creating a new Scrapy project
2.Writing a spider to crawl a site and extract data
3.Exporting the scraped data using the command line
4.Changing spider to recursively follow links
5.Using spider arguments
6.Load the scraped data into a SQLlite3 database

scrapy-mini-project
  scrapy.cfg
  tutorial
    __init__.py
    items.py
    middlewares.py
    pipelines.py
    settings.py
    spiders
      __init__.py


