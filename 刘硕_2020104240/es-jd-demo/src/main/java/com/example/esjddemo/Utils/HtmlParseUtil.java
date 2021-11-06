package com.example.esjddemo.Utils;

import com.example.esjddemo.pojo.Content;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * @author liushuo
 */
public class HtmlParseUtil {
    public static void main(String[] args) throws Exception {
        new HtmlParseUtil().parseJD("心理学").forEach(System.out::println);
    }
    public List<Content> parseJD(String keywords) throws Exception {
        //获取请求
        String url = "https://search.jd.com/search?keyword="+keywords;
        //解析网页
        Document document = Jsoup.parse(new URL(url), 30000);
        Element element = document.getElementById("J_goodsList");
        //获取所有的li元素
        Elements elements = element.getElementsByTag("li");
        ArrayList<Content> goodsList = new ArrayList<>();

        for (Element el:elements) {
            String img = el.getElementsByTag("img").eq(0).attr("data-lazy-img");
            String price = el.getElementsByClass("p-price").eq(0).text();
            String title = el.getElementsByClass("p-name").eq(0).text();
            Content content = new Content();
            content.setTitle(title);
            content.setImg(img);
            content.setPrice(price);
            goodsList.add(content);
        }
        return goodsList;
    }

}
