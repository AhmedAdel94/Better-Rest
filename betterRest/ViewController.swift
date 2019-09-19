//
//  ViewController.swift
//  betterRest
//
//  Created by Ahmed Adel on 9/19/19.
//  Copyright © 2019 Ahmed Adel. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    var wakeupTime : UIDatePicker!
    
    var sleepAmountTime:UIStepper!
    var sleepAmountLabel:UILabel!
    
    var coffeeAmountStepper:UIStepper!
    var coffeeAmountLabel:UILabel!
    
    override func loadView() {
        view = UIView()
        view.backgroundColor = .white
        
        let mainStackView = UIStackView()
        mainStackView.axis = .vertical
        mainStackView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(mainStackView)
        
        NSLayoutConstraint.activate([
            mainStackView.topAnchor.constraint(equalTo: view.layoutMarginsGuide.topAnchor, constant: 20),
            mainStackView.leadingAnchor.constraint(equalTo: view.layoutMarginsGuide.leadingAnchor),
            mainStackView.trailingAnchor.constraint(equalTo: view.layoutMarginsGuide.trailingAnchor)
        ])
        
        let wakeUpTitle = UILabel()
        wakeUpTitle.font = UIFont.preferredFont(forTextStyle: .headline)
        wakeUpTitle.numberOfLines = 0
        wakeUpTitle.text = "When do you want to wake up?"
        mainStackView.addArrangedSubview(wakeUpTitle)
        
        wakeupTime = UIDatePicker()
        wakeupTime.datePickerMode = .time
        wakeupTime.minuteInterval = 15
        mainStackView.addArrangedSubview(wakeupTime)
        
        var components = Calendar.current.dateComponents([.hour,.minute], from: Date())
        components.hour = 8
        components.minute = 0
        wakeupTime.date = Calendar.current.date(from: components) ?? Date()
        
        let sleepTitle = UILabel()
        sleepTitle.font = UIFont.preferredFont(forTextStyle: .headline)
        sleepTitle.numberOfLines = 0
        sleepTitle.text = "What is the minimum amount of sleep you need?"
        mainStackView.addArrangedSubview(sleepTitle)
        
        sleepAmountTime = UIStepper()
        sleepAmountTime.addTarget(self, action: #selector(sleepAmountChange), for: .valueChanged)
        sleepAmountTime.stepValue = 0.25
        sleepAmountTime.value = 8
        sleepAmountTime.minimumValue = 4
        sleepAmountTime.maximumValue = 12
        
        sleepAmountLabel = UILabel()
        sleepAmountLabel.font = UIFont.preferredFont(forTextStyle: .body)
        
        let sleepStackView = UIStackView()
        sleepStackView.spacing = 20
        sleepStackView.addArrangedSubview(sleepAmountTime)
        sleepStackView.addArrangedSubview(sleepAmountLabel)
        
        mainStackView.addArrangedSubview(sleepStackView)
        
        let coffeeTitle = UILabel()
        coffeeTitle.font = UIFont.preferredFont(forTextStyle: .headline)
        coffeeTitle.text = "How much coffee you drink by day?"
        coffeeTitle.numberOfLines = 0
        mainStackView.addArrangedSubview(coffeeTitle)
        
        coffeeAmountStepper = UIStepper()
        coffeeAmountStepper.addTarget(self, action: #selector(coffeeAmountChanged), for: .valueChanged)
        coffeeAmountStepper.minimumValue = 1
        coffeeAmountStepper.maximumValue = 20
//        coffeeAmountStepper.stepValue = 1
        
        coffeeAmountLabel = UILabel()
        coffeeAmountLabel.font = UIFont.preferredFont(forTextStyle: .body)
        
        let coffeeStackView = UIStackView()
        coffeeStackView.spacing = 20
        coffeeStackView.addArrangedSubview(coffeeAmountStepper)
        coffeeStackView.addArrangedSubview(coffeeAmountLabel)
        mainStackView.addArrangedSubview(coffeeStackView)
        
        mainStackView.setCustomSpacing(10, after: sleepTitle)
        mainStackView.setCustomSpacing(20, after: sleepStackView)
        mainStackView.setCustomSpacing(10, after: coffeeTitle)
        
        sleepAmountChange()
        coffeeAmountChanged()
        
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Better Rest"
        navigationController?.navigationBar.prefersLargeTitles = true
        navigationItem.rightBarButtonItem = UIBarButtonItem(title: "Calculate", style: .plain, target: self, action: #selector(calculateBedTime))
    }
    
    @objc func sleepAmountChange()
    {
        sleepAmountLabel.text = String(format:"%g hours" , sleepAmountTime.value)
    }
    
    @objc func coffeeAmountChanged()
    {
        if coffeeAmountStepper.value == 1{
            coffeeAmountLabel.text = "1 cup"
        }else{
            coffeeAmountLabel.text = "\(Int(coffeeAmountStepper.value)) cups"
        }
    }
    
    @objc func calculateBedTime()
    {
        let model = sleepCalculator()
        
        let title:String
        let message:String
        
        do{
            let components = Calendar.current.dateComponents([.hour,.minute], from: wakeupTime.date)
            let hour = (components.hour ?? 0) * 60 * 60
            let minutes = (components.minute ?? 0)*60
            
            let prediction = try model.prediction(coffee: coffeeAmountStepper.value, estimatedSleep: sleepAmountTime.value, wake: Double(hour+minutes))
            
            let formatter = DateFormatter()
            formatter.timeStyle = .short
            
            let wakeTime = wakeupTime.date - prediction.actualSleep
            message = formatter.string(from: wakeTime)
            title = "Your ideal bed time is"
            
        }catch{
            title = "Error"
            message = "There was an error calculating your bedtime"
        }
        
        let ac = UIAlertController(title: title, message: message, preferredStyle: .alert)
        ac.addAction(UIAlertAction(title: "OK", style: .default))
        present(ac,animated: true)

    }
    


}

