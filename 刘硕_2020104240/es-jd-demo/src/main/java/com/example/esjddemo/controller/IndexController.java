package com.example.esjddemo.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

/**
 * @author liushuo
 */
@Controller
public class IndexController {
    @GetMapping({"/","/index"})
    public String index(){
        return "index";
    }
}
