import scrapy


class QuotesSpider(scrapy.Spider):
    name = 'toscrape-xpath'

    start_urls = ['http://quotes.toscrape.com']

    def parse(self, response):
        for quote in response.xpath('//div[@class="quote"]'):
            yield {
                'text': quote.xpath('span[@class="text"]/text()').extract(),
                'author': quote.xpath('span/small/text()').extract(),
                'tags': quote.xpath('div[@class="tags"]/a[@class="tag"]/text()').extract(),
            }

        next_page = response.xpath('//li[@class="next"]/a')[0]
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)